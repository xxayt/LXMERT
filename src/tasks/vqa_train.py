# coding=utf-8
# Copyleft 2019 project LXRT.

import os

import torch
import time
import datetime
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timm.utils import accuracy, AverageMeter
# 
from param import args
from utils import *
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import get_data_tuple

SHOWITER_TRAIN_BASE_NUM = 1500 # 19753  # bs=32
SHOWITER_EVAL_BASE_NUM = 3 # 26  # bs=1024
SHOWITER_TRAIN_TINY_NUM = 10 # 90
SHOWITER_EVAL_TINY_NUM = 1 # 3

def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = args.save_dir  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
    # tensorboard为acc和loss画图
    tb_writer = SummaryWriter(args.path_log)
    # logger.info args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get datasets
    train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True, logger=logger)
    eval_tuple = get_data_tuple(args.valid, bs=1024, shuffle=False, drop_last=False, logger=logger) if args.valid != "" else None
    # get net
    logger.info(f"Creating model: VQAModel")
    model = VQAModel(train_tuple.dataset.num_answers)

    # load pre-trained weights 加载预训练权重
    if args.load_lxmert_qa is not None:
        logger.info("Loading LXMERT QA weights from %s" % args.load_lxmert_qa)
        load_lxmert_qa(path=args.load_lxmert_qa, model=model, label2ans=train_tuple.dataset.label2ans, logger=logger)
    if args.load is not None:
        logger.info("Load VQA model from %s" % args.load)
        checkpoint = torch.load("%s" % args.load)['model']
        model_dict = model.state_dict()
        state_dict = {k.replace('module.', ''):v for k,v in checkpoint.items() if k in model_dict.keys()}  # load same name layer weiget
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)  # 载入我自己训练的vqa_large_base模型权重
        # 冻结后半部分权重  for exchange2
        unfreeze_layers = ['x_layers.5', 'x_layers.6', 'x_layers.7', 'x_layers.8', 'x_layers.9', 'x_layers.10', 'bert.pooler', 'logit_fc']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    # get criterion 损失函数
    loss_function = nn.BCEWithLogitsLoss()
    # get optimizer 优化器
    train_layer = [p for p in model.parameters() if p.requires_grad == True]  # 只优化需要更新的参数
    if 'bert' in args.optim:
        batch_per_epoch = len(train_tuple.loader)  # 90 or 19753
        t_total = int(batch_per_epoch * args.epochs)  # 90 * 4
        logger.info(f"Batch per epoch: {batch_per_epoch}, Total Iters: {t_total}")
        logger.info(f"BertAdam Total Iters: {t_total}")
        from lxrt.optimization import BertAdam
        optimizer = BertAdam(list(train_layer),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)
    else:
        optimizer = args.optimizer(train_layer, args.lr)

    model.cuda()
    model = torch.nn.DataParallel(model)

    start_epoch = 0
    max_accuracy = 0.0
    # 继续训练
    if args.resume:
        if args.resume in ['Best', 'Last']:
            args.resume = os.path.join(args.path_log, '%s-%s.pth' % (args.name, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            state_dict = torch.load(args.resume)
            if 'model' in state_dict:
                start_epoch = state_dict['epoch'] + 1
                model.load_state_dict(state_dict['model'],strict=False)
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
            else:
                model.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
            if 'max_accuracy' in state_dict:
                max_accuracy = state_dict['max_accuracy']
            val_loss, val_acc = validate(eval_tuple, model, loss_function, state_dict['epoch'], logger, args)
            max_accuracy = max(max_accuracy, val_acc)
            logger.info(f'Max accuracy: {max_accuracy:.4f}%')
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    # Train
    logger.info("Start training")
    best_acc1 = 0.0
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch_local_data(train_tuple, model, loss_function, optimizer, epoch, logger, args, tb_writer=tb_writer)
        save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='Last')
        # validate
        logger.info(f"**********Latest val***********")
        val_loss, val_acc = validate(eval_tuple, model, loss_function, epoch, logger, args, tb_writer=tb_writer)
        if val_acc > best_acc1:
            best_acc1 = val_acc
            save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='Best')
        logger.info('Exp path: %s' % args.path_log)
    # 总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch_local_data(train_tuple, model, loss_function, optimizer, epoch, logger, args, tb_writer=None):
    Dataset, train_loader, evaluator = train_tuple
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    quesid2ans = {}
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for iter, (ques_id, feats, boxes, sent, target) in enumerate(train_loader):
        # 将数据转移到GPU上
        feats = feats.cuda(non_blocking=True)
        boxes = boxes.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # 计算模型输出
        output = model(feats, boxes, sent)
        assert output.dim() == target.dim() == 2
        # 计算loss
        loss = loss_function(output, target)
        loss = loss * output.size(1) # *3129
        loss.backward()  # compute gradients based on current loss
        # 为解决梯度爆炸问题，设置梯度截断(设置梯度大小的上限)
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        # 更新参数
        optimizer.step()  # update parameters based on current gradients
        optimizer.zero_grad()  # clear gradients for next train
        
        # 计算准确率
        _, label = output.max(1)  # dim=1按行取最大
        for qid, ans_index in zip(ques_id, label.cpu().numpy()):
            # 将预测答案index转换为ans
            ans = Dataset.label2ans[ans_index]
            # 构建quesid2ans字典: question_id作为key，预测答案作为value
            quesid2ans[qid.item()] = ans
        train_score = evaluator.evaluate(quesid2ans) * 100.

        # 储存batch_time和loss
        batch_time.update(time.time() - end)  # 记录每次迭代batch所需时间
        end = time.time()
        loss_meter.update(loss.item(), output.size(0))  # output.size(0)
        score_meter.update(train_score, output.size(0))  # output.size(0) = 3129??

        # log输出训练参数
        train_interval = SHOWITER_TRAIN_TINY_NUM if args.tiny else SHOWITER_TRAIN_BASE_NUM 
        if iter % train_interval == 0:
            etas = batch_time.avg * (num_steps - iter)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{iter}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'train_score {score_meter.val:.4f} ({score_meter.avg:.4f})\t')
        
        # tensorboard记录训练参数
        if tb_writer is not None:
            tags = ["train_loss", "train_acc"]
            tb_writer.add_scalar(tags[0], loss.item(), epoch * num_steps + iter)
            tb_writer.add_scalar(tags[1], train_score, epoch * num_steps + iter)
    
    logger.info(f"Train avg Score: {score_meter.avg}")
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, score_meter.avg

@torch.no_grad()
def validate(eval_tuple, model, loss_function, epoch, logger, args, dump=None, tb_writer=None):
    logger.info('eval epoch {}'.format(epoch))
    Dataset, val_loader, evaluator = eval_tuple
    model.eval()
    
    num_steps = len(val_loader)
    quesid2ans = {}
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    end = time.time()
    for iter, (ques_id, feats, boxes, sent, target) in enumerate(val_loader):
        with torch.no_grad():
            # 将数据转移到GPU上
            feats = feats.cuda(non_blocking=True)
            boxes = boxes.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # 计算模型输出
            output = model(feats, boxes, sent)

            loss = loss_function(output, target)
            loss = loss * output.size(1) # *3129
            # 计算准确率
            _, label = output.max(1)  # dim=1按行取最大
            for qid, ans_index in zip(ques_id, label.cpu().numpy()):
                ans = Dataset.label2ans[ans_index]
                quesid2ans[qid.item()] = ans
            eval_score = evaluator.evaluate(quesid2ans) * 100.

        # 更新记录
        loss_meter.update(loss.item(), target.size(0))
        score_meter.update(eval_score, target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # log输出测试参数
        eval_interval = SHOWITER_EVAL_TINY_NUM if args.tiny else SHOWITER_EVAL_BASE_NUM 
        if iter % eval_interval == 0:
            logger.info(
                f'Test: [{iter}/{num_steps}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'eval_score {score_meter.val:.4f} ({score_meter.avg:.4f})\t')
        
        # tensorboard记录测试参数
        if tb_writer is not None:
            tags = ["eval_loss", "eval_acc"]
            tb_writer.add_scalar(tags[0], loss.item(), epoch * num_steps + iter)
            tb_writer.add_scalar(tags[1], eval_score, epoch * num_steps + iter)
    
    if dump is not None:
        evaluator.dump_result(quesid2ans, dump)
    logger.info(f"Eval avg Score: {score_meter.avg}")
    return loss_meter.avg, eval_score

if __name__ == "__main__":
    main(args)