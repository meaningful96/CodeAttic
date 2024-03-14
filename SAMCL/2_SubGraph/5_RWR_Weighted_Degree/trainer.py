import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from randomwalk import RandomWalk, build_graph, build_graph_mapped, Path_Dictionary, Making_Subgraph, mapping_dict

from doc import Dataset, collate, load_data, Custom_Dataset
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from config import args

import numpy as np
import multiprocessing
import random
import time
import datetime
import pickle

"""
def load_dict_with_tuple_keys(file_name):
    # Load the .npy file
    loaded_dict = np.load(file_name, allow_pickle=True).item()

    # Convert string keys back to tuples
    return {tuple(map(int, k.split('_'))): v for k, v in loaded_dict.items()}
"""

def load_pkl(path):
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def mean_tensor(matrix):
    row_averages = torch.mean(matrix, dim=0)
    return row_averages


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
        
        self.train_loss = []
        self.valid_loss = []
        
        """
        self.train_entMapping_dict, self.train_relMapping_dict, _ = mapping_dict(args.train_path)
        self.valid_entMapping_dict, self.valid_relMapping_dict, _ = mapping_dict(args.valid_path)
        """

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction = 'none').cuda() # tail degree -> column 
        # self.criterion = nn.CrossEntropyLoss().cuda() # head degree -> row

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_obj, valid_obj = RandomWalk(args.train_path), RandomWalk(args.valid_path)
        dataset_length = len(json.load(open(args.train_path, 'r', encoding='utf-8')))
        valid_length = len(json.load(open(args.valid_path, 'r', encoding='utf-8')))
        logger.info("Train dataset length: {}".format(dataset_length))
        logger.info("Valid dataset length: {}".format(valid_length))
       
        self.train_Graph, self.train_Graph_tail, self.train_diGraph, self.train_appearance, self.train_entities = build_graph(args.train_path)
        self.valid_Graph, self.valid_Graph_tail, self.valid_diGraph, self.valid_appearance, self.valid_entities = build_graph(args.valid_path)

        """
        # Start to Random Walk
        # Only need to implement the randomwalk algorithm at once
        # Because all the paths are stored in the `train_data_dict'
        # We just make subgraphs in every epoch.

        # train_data_dict
        # {(h, r, t):[[path1], [path2],...], (h2,r2,t2):{[path1], [path2],...} }
        # The keys such as (h, r, t) are tuples. tuple(str, str, str)
        """

        self.batch_size = args.batch_size
        self.subgraph_size = args.subgraph_size
        self.step_size = (dataset_length*2) // args.batch_size
        self.valid_step_size = (valid_length*2) // args.batch_size
        
        num_training_steps = args.epochs * self.step_size
        
        self.batch_duplication, self.sub_duplication = [], []

        self.train_num_candidates = self.step_size * (self.batch_size) //(self.subgraph_size*2) 
        self.valid_num_candidates = self.valid_step_size * (self.batch_size) // (args.subgraph_size*2)
        args.warmup = min(args.warmup, num_training_steps // 10)

        logger.info("Step Size per Epoch: {}".format(self.step_size))
        logger.info("Training Candidates: {}".format(self.train_num_candidates))
        logger.info("Validation Candidates: {}".format(self.valid_num_candidates))
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))

        # Random Walk
        logger.info("Start RandomWalk!!")
        
        # If you want to run the code without creating an npy file, add the lines below
        """
        self.train_data_dict = Path_Dictionary(args.train_path, self.k_steps, self.num_iter, train_obj, args.num_process, args.subgraph_size)
        self.train_initial_triples = random.sample(list(self.train_data_dict.keys()), self.train_num_candidates)
        
        self.valid_data_dict = Path_Dictionary(args.valid_path, self.k_steps, self.num_iter, valid_obj, args.num_process, args.subgraph_size)       
        self.valid_initial_triples = random.sample(list(self.valid_data_dict.keys()), self.valid_num_candidates)
        """
        
        # Path Dictionary
        # train_data_dict = load_dict_with_tuple_keys(args.train_path_dict)
        # valid_data_dict = load_dict_with_tuple_keys(args.valid_path_dict)
        # train_data_dict = np.load(args.train_path_dict, allow_pickle=True).item()
        # valid_data_dict = np.load(args.valid_path_dict, allow_pickle=True).item()
        
        self.train_data_dict = load_pkl(args.train_path_dict)
        self.valid_data_dict = load_pkl(args.valid_path_dict)

        # Inintial triples
        self.train_initial_triples = random.sample(list(self.train_data_dict.keys()), self.train_num_candidates)
        self.valid_initial_triples = random.sample(list(self.valid_data_dict.keys()), self.valid_num_candidates)

        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None
        self.validation = args.validation

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        train_appearance = self.train_appearance
        valid_appearance = self.valid_appearance

        start_train = time.time()
        appearance_train = train_appearance
        appearance_valid = valid_appearance
        for epoch in range(self.args.epochs):
            start_epoch = time.time()
            start_selection = time.time()
            logger.info("Making Subgraph!!")
            ss = time.time()
            if epoch == 0:
                candidates_train = self.train_initial_triples
                candidates_valid = self.valid_initial_triples

                train_data, counted_appearance_train, sub_total, batch_total  = Making_Subgraph(self.train_data_dict,
                                                                                                # self.train_entMapping_dict,
                                                                                                # self.train_relMapping_dict,
                                                                                                candidates_train, 
                                                                                                self.subgraph_size, 
                                                                                                appearance_train, 
                                                                                                self.batch_size)
                
                valid_data, counted_appearance_valid, _, _  = Making_Subgraph(self.valid_data_dict,
                                                                              # self.valid_entMapping_dict,
                                                                              # self.valid_relMapping_dict,
                                                                              candidates_valid, 
                                                                              self.subgraph_size, 
                                                                              appearance_valid, 
                                                                              self.batch_size)    
            
                self.batch_duplication.extend(batch_total)
                self.sub_duplication.extend(sub_total)

                start_count = time.time()
                sorted_candidates_train = sorted(counted_appearance_train.items(), key=lambda x: x[1]) # key-value. x[1] -> sorted by value
                new_candidates_train = sorted_candidates_train[:self.train_num_candidates]
                candidates_train = [item[0] for item in new_candidates_train]

                sorted_candidates_valid = sorted(counted_appearance_valid.items(), key=lambda x: x[1]) # key-value. x[1] -> sorted by value
                new_candidates_valid = sorted_candidates_valid[:self.valid_num_candidates]
                candidates_valid = [item[0] for item in new_candidates_valid]           
                end_count = time.time()

                print("Time for Counting = '{}'".format(datetime.timedelta(seconds = end_count - start_count)))    
                
                appearance_train = counted_appearance_train
                appearance_valid = counted_appearance_valid

            else:
                train_data, counted_appearance_train, sub_total, batch_total  = Making_Subgraph(self.train_data_dict,
                                                                                                # self.train_entMapping_dict,
                                                                                                # self.train_relMapping_dict,
                                                                                                candidates_train, 
                                                                                                self.subgraph_size, 
                                                                                                appearance_train, 
                                                                                                self.batch_size)
                
                valid_data, counted_appearance_valid, _, _  = Making_Subgraph(self.valid_data_dict,
                                                                              # self.valid_entMapping_dict,
                                                                              # self.valid_relMapping_dict,
                                                                              candidates_valid, 
                                                                              self.subgraph_size, 
                                                                              appearance_valid, 
                                                                              self.batch_size)      
                self.batch_duplication.extend(batch_total)
                self.sub_duplication.extend(sub_total)

                start_count = time.time()
                sorted_candidates_train = sorted(counted_appearance_train.items(), key=lambda x: x[1]) # key-value. x[1] -> sorted by value
                new_candidates_train = sorted_candidates_train[:self.train_num_candidates]
                candidates_train = [item[0] for item in new_candidates_train]
            
                sorted_candidates_valid = sorted(counted_appearance_valid.items(), key=lambda x: x[1]) # key-value. x[1] -> sorted by value
                new_candidates_valid = sorted_candidates_valid[:self.valid_num_candidates]
                candidates_valid = [item[0] for item in new_candidates_valid]       
                end_count = time.time()

                print("Time for Counting = '{}'".format(datetime.timedelta(seconds = end_count - start_count)))    
                
                appearance_train = counted_appearance_train
                appearance_valid = counted_appearance_valid

            ee = time.time()
            logger.info("Time for building Subgraphs: {}".format(datetime.timedelta(seconds = ee - ss)))
            train_dataset = Custom_Dataset(data=train_data)
            valid_dataset = Custom_Dataset(data=valid_data)
        
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

            end_selection = time.time()
            logger.info("Time for Making Subgraphs and Data Loading: {}".format(datetime.timedelta(seconds = end_selection - start_selection)))
                          
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
        
        batch_meanDuple = sum(self.batch_duplication) / len(self.batch_duplication)
        sub_meanDuple = sum(self.sub_duplication) / len(self.sub_duplication)
        print("Mean of Subgraph Duplication: {}".format(sub_meanDuple))
        print("Mean of Batch Duplication: {}".format(batch_meanDuple))
        
        # Counting the triples that never used
        zero_count = 0
        for value in list(appearance_train.values()):
            if value == 0:
                zero_count += 1
        print("Never Used Triplets: {}".format(zero_count))

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

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            degree_tail = outputs.degree_tail
            logits, labels = outputs.logits, outputs.labels
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
        self.valid_loss.append(round(losses.avg, 3))
        print("Valid_loss = {}".format(self.valid_loss))
        return metric_dict

    
    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        inv_b = AverageMeter('InvB', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, inv_b, top1, top3],
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
            logits, labels = outputs.logits, outputs.labels
            degree_head, degree_tail = outputs.degree_head, outputs.degree_tail

            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss_forward = self.criterion(logits, labels) * degree_tail
            loss = mean_tensor(loss_forward)
            # loss_forward = mean_tensor(loss_forward).to(logits.device)
            # tail -> head + relation
            loss_backward = self.criterion(logits[:, :batch_size].t(), labels) * degree_head

            # loss_backward = mean_tensor(loss_backward).to(logits.device)
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
        self.train_loss.append(round(losses.avg, 3))
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
