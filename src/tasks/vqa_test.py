# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

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

SHOWITER_TRAIN_BASE_NUM = 1500 # 19753
SHOWITER_EVAL_BASE_NUM = 3 # 26
SHOWITER_TRAIN_TINY_NUM = 10 # 90
SHOWITER_EVAL_TINY_NUM = 1 # 3


def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = args.save_dir  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))
    # get tiny train datasets
    train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True, logger=logger)  # tiny=True使得train_tuple加载速度很快

    # get net
    logger.info(f"Creating model: VQAModel")
    model = VQAModel(train_tuple.dataset.num_answers)  # train_tuple.dataset.num_answers=3129
    # get criterion 损失函数
    loss_function = nn.BCEWithLogitsLoss()

    # load VQA weights
    if args.load is not None:
        logger.info("Load VQA model from %s" % args.load)
        checkpoint = torch.load("%s" % args.load)
        # 模型保存加入了'module.'前缀，需要去掉
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
    
    model.cuda()
    model = torch.nn.DataParallel(model)

    # Test
    logger.info(f"Start testing")
    args.fast = args.tiny = False       # Always loading all data in test
    if 'test' in args.test:
        test_tuple = get_data_tuple(args.test, bs=950, shuffle=False, drop_last=False, logger=logger)
        dump_path = os.path.join(args.path_log, args.load.split('/')[-1][:-4] + '-test_predict.json')
        logger.info(f"Dump path: {dump_path}")
        test(test_tuple, model, 0, logger, args, dump=dump_path)
    elif 'val' in args.test:
        test_tuple = get_data_tuple('minival', bs=950, shuffle=False, drop_last=False, logger=logger)
        dump_path = os.path.join(args.path_log, 'minival_predict.json')
        validate(test_tuple, model, loss_function, 0, logger, args, dump=dump_path)
    else:
        assert False, "No such test option for %s" % args.test

@torch.no_grad()
def validate(test_tuple, model, loss_function, epoch, logger, args, dump=None):
    logger.info('eval epoch {}'.format(epoch))
    Dataset, val_loader, evaluator = test_tuple
    model.eval()
    
    quesid2ans = {}
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    end = time.time()
    for iter, data in enumerate(val_loader):
        with torch.no_grad():
            # ques_id, feats, boxes, sent, target = data
            ques_id, feats, boxes, sent = data[:4]   # Avoid seeing ground truth
            # 将数据转移到GPU上
            feats = feats.cuda(non_blocking=True)
            boxes = boxes.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            # 计算模型输出
            output = model(feats, boxes, sent)

            # loss = loss_function(output, target)
            # loss = loss * output.size(1) # *3129
            # 计算准确率
            _, label = output.max(1)  # dim=1按行取最大
            for qid, ans_index in zip(ques_id, label.cpu().numpy()):
                ans = Dataset.label2ans[ans_index]
                quesid2ans[qid.item()] = ans
            eval_score = evaluator.evaluate(quesid2ans) * 100.

        # 更新记录
        # loss_meter.update(loss.item(), target.size(0))
        score_meter.update(eval_score, 3129)  # target.size(0)
        batch_time.update(time.time() - end)
        end = time.time()
        # log输出测试参数
        eval_interval = SHOWITER_EVAL_TINY_NUM if args.tiny else SHOWITER_EVAL_BASE_NUM 
        if iter % eval_interval == 0:
            logger.info(
                f'Test: [{iter}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'eval score {score_meter.val:.4f} ({score_meter.avg:.4f})\t')
    if dump is not None:
        evaluator.dump_result(quesid2ans, dump)
    logger.info(f"Eval avg Score: {score_meter.avg}")
    return eval_score

@torch.no_grad()
def test(test_tuple, model, epoch, logger, args, dump=None):
    logger.info('eval epoch {}'.format(epoch))
    Dataset, val_loader, evaluator = test_tuple
    model.eval()
    
    quesid2ans = {}
    for iter, data in enumerate(val_loader):
        with torch.no_grad():
            ques_id, feats, boxes, sent = data[:4]   # Avoid seeing ground truth
            # 将数据转移到GPU上
            feats = feats.cuda(non_blocking=True)
            boxes = boxes.cuda(non_blocking=True)
            # 计算模型输出
            output = model(feats, boxes, sent)
            # 计算准确率
            _, label = output.max(1)  # dim=1按行取最大
            for qid, ans_index in zip(ques_id, label.cpu().numpy()):
                ans = Dataset.label2ans[ans_index]
                quesid2ans[qid.item()] = ans
            if iter % 50 == 0:
                logger.info(f"Eval {iter}/{len(val_loader)} finished")
    # 将预测结果保存到json文件中
    if dump is not None:
        logger.info(f"Dump result to {dump}")
        evaluator.dump_result(quesid2ans, dump)

if __name__ == "__main__":
    main(args)