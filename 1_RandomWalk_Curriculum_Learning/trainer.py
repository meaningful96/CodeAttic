import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Custom_Dataset, collate, load_data, Dataset
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from randomwalk import RandomWalk, random_walk_for_example, process_data, randomwalk_sampling
from config import args

import time
import datetime

import random

def shuffle_dataset(data, group_size):
    total_items = len(data)
    num_groups = total_items // group_size
    # Split the data into groups
    grouped_data = [data[i:i+group_size] for i in range(0, total_items, group_size)]

    # Shuffle the order of the groups
    random.shuffle(grouped_data)

    # Flatten the list of shuffled groups
    shuffled_data_accumulated = [item for sublist in grouped_data for item in sublist]

    return shuffled_data_accumulated

class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.task = args.task
        self.batch_size = args.batch_size
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)
        
        self.train_loss, self.valid_loss = [], []
        self.negative_size = args.negative_size
        
        self.train_dataset = json.load(open(args.train_path, 'r', encoding='utf-8'))
        self.valid_dataset = json.load(open(args.valid_path, 'r', encoding='utf-8'))
        
        self.G_train = RandomWalk(args.train_path)
        self.G_valid = RandomWalk(args.valid_path)

        num_training_steps = args.epochs * len(self.train_dataset) * self.negative_size *2 // args.batch_size
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None
        self.validation = args.validation

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
       
        start_train = time.time()
        if self.task = "wn18rr":
            negative = 2
        if self.task = 'fb15k237':
            negative = 16
        count = 0 
        for epoch in range(self.args.epochs):
            start_epoch = time.time()
            
            start_sh = time.time()
            if epoch == 0:
                args.cur = True
                args.use_self_negative = True

                train_dataset = Dataset(path=args.train_path, task=args.task)
                valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
                self.train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=True)

                self.valid_loader = None
                if valid_dataset:
                    self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=2*args.batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    num_workers=args.workers,
                    pin_memory=True) 
                logger.info("Curriculum Learning!! Epoch {}".format(epoch))

            if epoch >= 1:
                if epoch == 1:
                    neg = negative
                if epoch > 1:
                    neg = negative * 2
                args.cur = False
                sampling_train = randomwalk_sampling(self.train_dataset, self.G_train, neg)
                # sampling_valid = randomwalk_sampling(self.valid_dataset, self.G_valid, neg)

                train_data = shuffle_dataset(sampling_train, self.negative_size * 2)
                valid_data = shuffle_dataset(self.valid_dataset, 1)
            
                logger.info("Random Walk Sampling + Group Shuffling Done!!")
            
                shuffled_train_dataset = Custom_Dataset(data=train_data)
                shuffled_valid_dataset = Custom_Dataset(data=valid_data)
                self.train_loader = torch.utils.data.DataLoader(
                    shuffled_train_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=self.args.workers,
                    pin_memory=True,
                    drop_last=True)
        
                self.valid_loader = None
                if self.valid_dataset:
                    self.valid_loader = torch.utils.data.DataLoader(
                        shuffled_valid_dataset,
                        batch_size=2*self.batch_size,
                        shuffle=False,
                        collate_fn=collate,
                        num_workers=self.args.workers,
                        pin_memory=True,
                        drop_last=True)
            
                end_sh = time.time()
                logger.info("Time for RandomWalk Sampling + Group Shuffling + Loading the data: {}".format(datetime.timedelta(seconds = end_sh - start_sh)))
                

            # train for one epoch
            self.train_epoch(epoch)
            # validation step
            args.validation = True
            self._run_eval(epoch=epoch)
            end_epoch = time.time()
            print("Time_per_Epoch = '{}'".format(datetime.timedelta(seconds = end_epoch - start_epoch)))
            args.validation = False
        end_train = time.time()
        print("Total_Training_Time = '{}'".format(datetime.timedelta(seconds = end_train - start_train)))

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        args.validation=True
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])
            
            if args.cur:
                outputs = self.model(**batch_dict)
                outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
                outputs = ModelOutput(**outputs)
                logits, labels = outputs.logits, outputs.labels
                loss = self.criterion(logits, labels)
                losses.update(loss.item(), batch_size)

                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            if not args.cur:
                outputs = self.model(**batch_dict)
                outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
                outputs = ModelOutput(**outputs)
                logits_hr, logits_tail, labels_hr = outputs.logits_hr, outputs.logits_tail, outputs.labels_hr
                loss = self.criterion(logits_hr, labels_hr)
                losses.update(loss.item(), batch_size)

                acc1, acc3 = accuracy(logits_hr, labels_hr, topk=(1, 3))

            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        self.valid_loss.append(round(losses.avg, 3))
        print("Valid_loss = {}".format(self.valid_loss))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        inv_tt = AverageMeter('InvTT', '"6.2f"')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            
            if args.cur:
                logits, labels = outputs.logits, outputs.labels
                # head + relation -> tail
                loss = self.criterion(logits, labels)
                # tail -> head + relation
                loss += self.criterion(logits[:, :batch_size].t(), labels)

                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)

                inv_t.update(outputs.inv_t, 1)
                losses.update(loss.item(), batch_size)

            if not args.cur:
                logits_hr, logits_tail, labels_hr, labels_tail = outputs.logits_hr, outputs.logits_tail, outputs.labels_hr, outputs.labels_tail
                logits_reverse = outputs.logits_reverse

                # head + relation -> tail
                loss = self.criterion(logits_hr, labels_hr)
                # tail -> head + relation
                loss += self.criterion(logits_tail, labels_tail)
                # reverse triplet
                loss += self.criterion(logits_reverse, labels_tail)
                acc1, acc3 = accuracy(logits_hr, labels_hr, topk=(1, 3))

                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)

                inv_t.update(outputs.inv_t, 1)
                inv_tt.update(outputs.inv_t_hard, 1)
                losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        self.train_loss.append(round(losses.avg,3))
        print("Train_loss = {}".format(self.train_loss))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids = [0,1,2,3,4]).to("cuda:0")

        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
