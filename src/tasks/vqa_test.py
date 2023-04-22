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
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
# 创建具有命名字段的 tuple 子类的 factory 函数 (具名元组)
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

SHOWITER_TRAIN_BASE_NUM = 1500 # 19753
SHOWITER_EVAL_BASE_NUM = 3 # 26
SHOWITER_TRAIN_TINY_NUM = 10 # 90
SHOWITER_EVAL_TINY_NUM = 1 # 3

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    Dataset = VQADataset(splits)
    tset = VQATorchDataset(Dataset)
    evaluator = VQAEvaluator(Dataset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=Dataset, loader=data_loader, evaluator=evaluator)

def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = args.save_dir  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get datasets
    train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)
    if 'test' in args.test:
        test_tuple = get_data_tuple(args.test, bs=950, shuffle=False, drop_last=False)
        dump_path = os.path.join(args.path_log, 'test_predict.json')
    elif 'val' in args.test:
        test_tuple = get_data_tuple('minival', bs=950, shuffle=False, drop_last=False)
        dump_path = os.path.join(args.path_log, 'minival_predict.json')
    else:
        assert False, "No such test option for %s" % args.test
    # get net
    logger.info(f"Creating model: VQAModel")
    print("train_tuple.dataset.num_answers: ", train_tuple.dataset.num_answers)
    model = VQAModel(train_tuple.dataset.num_answers)
    # get criterion 损失函数
    loss_function = nn.BCEWithLogitsLoss()

    # load VQA weights
    if args.load is not None:
        logger.info("Load VQA model from %s" % args.load)
        state_dict = torch.load("%s.pth" % args.load)
        model.load_state_dict(state_dict)
    
    model.cuda()
    model = torch.nn.DataParallel(model)

    # Test
    logger.info(f"Start testing")
    args.fast = args.tiny = False       # Always loading all data in test
    validate(test_tuple, model, loss_function, 0, logger, args, dump=dump_path)

@torch.no_grad()
def validate(eval_tuple, model, loss_function, epoch, logger, args, dump=None):
    logger.info('eval epoch {}'.format(epoch))
    Dataset, val_loader, evaluator = eval_tuple
    model.eval()
    
    quesid2ans = {}
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
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
        batch_time.update(time.time() - end)
        end = time.time()
        # log输出测试参数
        eval_interval = SHOWITER_EVAL_TINY_NUM if args.tiny else SHOWITER_EVAL_BASE_NUM 
        if iter % eval_interval == 0:
            logger.info(
                f'Test: [{iter}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'eval score {eval_score})\t')
    if dump is not None:
        evaluator.dump_result(quesid2ans, dump)
    logger.info(f"Eval Score: {eval_score}")
    return loss_meter.avg, eval_score

if __name__ == "__main__":
    main(args)