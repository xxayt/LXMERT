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
from timm.utils import AverageMeter
# 
from param import args
from utils import create_logging
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
# 创建具有命名字段的 tuple 子类的 factory 函数 (具名元组)
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


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


class VQA:
    def __init__(self, logger):
        # logger
        self.logger = logger
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.loss_function = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            self.logger.info(f"BertAdam Total Iters: {t_total}")
            from lxrt.optimization import BertAdam
            self.optimizer = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optimizer = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.save_dir
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        self.logger.info("Start training")
        start_time = time.time()

        Dataset, loader, evaluator = train_tuple
        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_acc1 = 0.
        start_time = time.time()
        for epoch in range(args.epochs):
            quesid2ans = {}
            for iter, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            # for iter, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optimizer.zero_grad()

                feats, boxes, target = feats.cuda(non_blocking=True), boxes.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output = self.model(feats, boxes, sent)
                assert output.dim() == target.dim() == 2
                loss = self.loss_function(output, target)
                loss = loss * output.size(1)

                loss.backward()  # compute gradients based on current loss
                # 为解决梯度爆炸问题，设置梯度截断(设置梯度大小的上限)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()  # update parameters based on current gradients

                score, label = output.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = Dataset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                val_acc = self.evaluate(eval_tuple)
                if val_acc > best_acc1:
                    best_acc1 = val_acc
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, val_acc * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_acc1 * 100.)
                log_str += "This epoth takes: Time %0.2f\n" % (time.time() - start_time)
                start_time = time.time()

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

        # 训练总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        Dataset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                output = self.model(feats, boxes, sent)
                score, label = output.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = Dataset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        Dataset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = Dataset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        self.logger.info("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = args.save_dir  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件

    # get datasets
    train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)
    valid_tuple = get_data_tuple(args.valid, bs=1024, shuffle=False, drop_last=False) if args.valid != "" else None

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get net
    logger.info(f"Creating model: VQAModel")
    model = VQAModel(train_tuple.dataset.num_answers)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # get criterion 损失函数
    loss_function = nn.BCEWithLogitsLoss()
    # get optimizer 优化器
    if 'bert' in args.optim:
        batch_per_epoch = len(train_tuple.loader)
        t_total = int(batch_per_epoch * args.epochs)
        print(f"Batch per epoch: {batch_per_epoch}, Total Iters: {t_total}")
        logger.info(f"BertAdam Total Iters: {t_total}")
        from lxrt.optimization import BertAdam
        optimizer = BertAdam(list(model.parameters()),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)
    else:
        optimizer = args.optimizer(model.parameters(), args.lr)

    # tensorboard为acc和loss画图
    tb_writer = SummaryWriter(args.path_log)

    # load pre-trained weights 加载预训练权重
    if args.load_lxmert is not None:
        # 不走这里 #
        print("Loading LXMERT weights from %s" % args.load_lxmert)
        model.lxrt_encoder.load(args.load_lxmert)
    if args.load_lxmert_qa is not None:
        # 走这里 #
        print("Loading LXMERT QA weights from %s" % args.load_lxmert_qa)
        load_lxmert_qa(path=args.load_lxmert_qa, model=model, label2ans=train_tuple.dataset.label2ans, logger=logger)
    # load VQA weights
    if args.load is not None:
        logger.info("Load VQA model from %s" % args.load)
        state_dict = torch.load("%s.pth" % args.load)
        model.load_state_dict(state_dict)
    
    # Train
    if args.test is None:
        # if valid_tuple is not None:
        #     logger.info('Splits in Valid data:', valid_tuple.dataset.splits)
        #     logger.info("Valid Oracle: %0.2f" % (vqa.oracle_score(valid_tuple) * 100))
        # else:
        #     logger.info("DO NOT USE VALIDATION")
        # vqa.train(train_tuple, valid_tuple)

        logger.info("Start training")
        best_acc1 = 0.0
        start_time = time.time()
        
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch_local_data(train_tuple, model, loss_function, optimizer, epoch, logger, args)
            save("LAST")
            # validate
            logger.info(f"**********Latest val***********")
            val_loss, val_acc = validate(valid_tuple, model, loss_function, epoch, logger, args)
            if val_acc > best_acc1:
                best_acc1 = val_acc
                save("BEST")
            logger.info('Exp path: %s' % args.path_log)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
    # Test
    else:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950, shuffle=False, drop_last=False),
                dump=os.path.join(args.save_dir, 'test_predict.json')
            )
        elif 'val' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=950, shuffle=False, drop_last=False),
                dump=os.path.join(args.save_dir, 'minival_predict.json')
            )
            logger.info(result)
        else:
            assert False, "No such test option for %s" % args.test

############################################################
    # Build Class
    vqa = VQA()

def train_one_epoch_local_data(train_tuple, model, loss_function, optimizer, epoch, logger, args):
    Dataset, train_loader, evaluator = train_tuple
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    quesid2ans = {}
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for iter, (ques_id, feats, boxes, sent, target) in enumerate(train_loader):
        feats = feats.cuda(non_blocking=True)
        boxes = boxes.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(feats, boxes, sent)
        assert output.dim() == target.dim() == 2
        loss = loss_function(output, target)
        loss = loss * output.size(1)
        print("output.size(1):", output.size(1))
        acc1, label = output.max(1)

        loss.backward()  # compute gradients based on current loss
        # 为解决梯度爆炸问题，设置梯度截断(设置梯度大小的上限)
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()  # update parameters based on current gradients
        optimizer.zero_grad()  # clear gradients for next train
        
        for qid, l in zip(ques_id, label.cpu().numpy()):
            ans = Dataset.label2ans[l]
            quesid2ans[qid.item()] = ans

        # 储存batch_time和loss
        batch_time.update(time.time() - end)  # 记录每次迭代batch所需时间
        end = time.time()
        loss_meter.update(loss.item(), output.size(0))  # output.size(0)
        acc1_meter.update(acc1.item(), output.size(0))

        # log输出训练参数
        if iter % 50 == 0:
            etas = batch_time.avg * (num_steps - iter)
            # lr = optimizer.param_groups[0]['lr']
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{iter}/{num_steps}]\t'
                f'train acc: {evaluator.evaluate(quesid2ans) * 100.}\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})')

@torch.no_grad()
def validate(val_loader, model, loss_function, epoch, logger, args):



if __name__ == "__main__":
    main(args)