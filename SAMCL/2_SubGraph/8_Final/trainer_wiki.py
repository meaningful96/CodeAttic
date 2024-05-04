import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from LKG_randomwalk import Biased_RandomWalk, Making_Subgraph_for_LKG

from doc import Dataset, collate, load_data, Custom_Dataset
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models_wiki import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from config import args

import numpy as np
import random
import time
import datetime
import pickle

def mean_tensor(matrix):
    row_averages = torch.mean(matrix, dim=0)
    return row_averages

def load_pkl(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        # logger.info(self.model)
        self._setup_training()
        self.k_steps = args.k_steps
        self.num_iter = args.num_iter
        self.subgraph_size = args.subgraph_size
        
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction = 'none').cuda() # tail degree -> column 

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)
        args.warmup = 400
        self.scheduler = None
        self.best_metric = None
        self.batch_size = args.batch_size 
        self.degree_train, self.degree_valid = load_pkl(args.degree_train), load_pkl(args.degree_valid)
        
    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        start_train = time.time()
        train_data_all = Making_Subgraph_for_LKG(self.args.train_path_dict)
        valid_data_all = Making_Subgraph_for_LKG(self.args.train_path_dict)
        step_size_train = len(train_data_all) * 2 //self.batch_size
        step_size_valid = len(valid_data_all) * 2 //self.batch_size

        logger.info("Total Triples for Training: {}".format(len(train_data_all)))
        logger.info("Total Triples for Valiation: {}".format(len(valid_data_all)))
        logger.info("Step Size per Epoch for Training: {}".format(step_size_train))
        logger.info("Step Size per Epoch for Training: {}".format(step_size_valid))
        
        num_training_steps = step_size_train * self.args.epochs

        self.scheduler = self._create_lr_scheduler(num_training_steps)

        for epoch in range(self.args.epochs):
            train_dataset = Custom_Dataset(data=train_data_all)
            valid_dataset = Custom_Dataset(data=valid_data_all)
        
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True)

            self.valid_loader = None
            if valid_dataset:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=True)
                          
            # train for one epoch
            self.train_epoch(epoch)

            # validation
            args.validation = True
            self._run_eval(epoch=epoch)
            args.validation = False
            end_epoch = time.time()
            print("Time_per_Epoch = '{}'".format(datetime.timedelta(seconds = end_epoch - start_epoch)))
        end_train = time.time()
        print("Total_Training_Time = '{}'".format(datetime.timedelta(seconds = end_train - start_train)))
              

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
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
        
        valid_loss = []
        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)

            degree_tail = self.degree_valid[1][i*args.batch_size:((i+1)*args.batch_size)]
            logits, labels = outputs.logits, outputs.labels
            degree_tail = torch.tensor(degree_tail).reshape(logits.size(0)).to(logits.device)

            loss = self.criterion(logits, labels) * degree_tail

            # tail degree
            loss = mean_tensor(loss).to(logits.device)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        valid_loss.append(round(losses.avg, 3))
        print("Valid_loss = {}".format(valid_loss))
        return metric_dict
    
    def train_epoch(self, epoch):
            # loss_backward = mean_tensor(loss_backward).to(logits.device)
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        inv_b = AverageMeter('InvB', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        train_loss = []
        for i, batch_dict in enumerate(self.train_loader):
            degree_head = self.degree_train[0][i*args.batch_size:((i+1)*args.batch_size)]
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
            # loss_backward = mean_tensor(loss_backward).to(logits.device)
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            degree_head = self.degree_train[0][i*args.batch_size:((i+1)*args.batch_size)]
            degree_tail = self.degree_train[1][i*args.batch_size:((i+1)*args.batch_size)]

            degree_head = torch.tensor(degree_head).reshape(logits.size(0)).to(logits.device)
            degree_tail = torch.tensor(degree_tail).reshape(logits.size(0)).to(logits.device)

            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss_forward = self.criterion(logits, labels) * degree_tail
            loss = mean_tensor(loss_forward)
            
            # tail -> head + relation
            loss_backward = self.criterion(logits[:, :batch_size].t(), labels) * degree_head

            loss += mean_tensor(loss_backward)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            inv_b.update(outputs.inv_b, 1)
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
        train_loss.append(round(losses.avg, 3))
        print("Train_loss = {}".format(train_loss))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids = [0,1,2,3,5,6]).to("cuda:0")
            # loss_backward = mean_tensor(loss_backward).to(logits.device)
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
