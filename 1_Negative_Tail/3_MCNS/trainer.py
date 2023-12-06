import json
import torch
import time
import datetime
import random
import networkx as nx

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate, load_data, make_negative_examples
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer, count_dict
from logger_config import logger

from negative import node_degree, negative_sampling 
from dfs import make_graph, DFS_PATH

class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()
        self.batch_size = args.batch_size
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        self.graph, self.directed_graph, train_len = make_graph(args.train_path)
        self.valid_graph, self.valid_directed_graph, valid_len = make_graph(args.valid_path)
        self.train_dfs = DFS_PATH(self.graph, self.directed_graph, args.walks_num)
        self.valid_dfs = DFS_PATH(self.valid_graph, self.valid_directed_graph, args.walks_num)

        train_pathes, self.train, self.candidates = load_data(args.train_path, self.graph, self.directed_graph, args.walks_num) 
        valid_pathes, self.valid, self.valid_candidates = load_data(args.valid_path, self.valid_graph, self.valid_directed_graph, args.walks_num)
        
        self.train_count_check, self.train_triplet_mapping = count_dict(args.train_path, train_pathes, args.walks_num)
        self.valid_count_check, self.valid_triplet_mapping = count_dict(args.valid_path, valid_pathes, args.walks_num)
                
        self.training_steps = train_len // max(args.batch_size//2, 1)
        self.valid_steps = valid_len // max(args.batch_size//2, 1)
        
        if self.candidates is None:
                logger.info('Start dfs to make train candidates dict')
                _, self.candidates = self.train_dfs.intermediate()
        if self.valid_candidates is None:
                logger.info('Start dfs to make valid candidates dict')
                _, self.valid_candidates = self.valid_dfs.intermediate()

        
        total_training_steps = args.epochs * self.training_steps
        args.warmup = min(args.warmup, total_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(total_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(total_training_steps)
        self.best_metric = None

        self.train_loader = None
        self.valid_loader = None

        self.train_loss = []
        self.valid_loss = []
        
        #check for shortest path
        self.shortest_pathes = []
        triplets = json.load(open(self.args.train_path, 'r', encoding='utf-8'))
        self.G = nx.Graph()
        for triple in triplets:
            head = triple['head_id']
            tail = triple['tail_id']
            self.G.add_edge(head, tail)
        
    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        batch_indices = None
        valid_batch_indices = None

        N_steps = self.args.N_steps
        N_negs = self.args.N_negs
        train_q1_dict = node_degree(self.graph)
        valid_q1_dict = node_degree(self.valid_graph)
        
        start_train = time.time() 
        # torch.cuda.set_device(5)
        for epoch in range(self.args.epochs):
            start_epoch = time.time()
            # train for one epoch
            batch_indices = self.train_epoch(epoch, batch_indices, N_steps, N_negs, train_q1_dict)
            valid_batch_indices = self._run_eval(epoch=epoch, N_steps=N_steps, N_negs=N_negs, valid_q1_dict=valid_q1_dict, valid_batch_indices = valid_batch_indices)
            end_epoch = time.time()
            logger.info(f'Time per Epoch: {datetime.timedelta(seconds = end_epoch - start_epoch)}')

        not_seen_train = 0
        not_seen_valid = 0

        for v in self.train_count_check.values():
            if v==0:
                not_seen_train +=1
        for v in self.valid_count_check.values():
            if v==0:
                not_seen_valid +=1
        
        logger.info('never seen in train: {}'.format(not_seen_train))
        logger.info('never seen in validation: {}'.format(not_seen_valid))
        logger.info('train_loss: {}'.format(self.train_loss))
        logger.info('val_loss: {}'.format(self.valid_loss))
        logger.info(self.shortest_pathes)
        logger.info(f'avg shortest path per epoch : {sum(self.shortest_pathes)/len(self.shortest_pathes)}')
        end_train = time.time()
        logger.info(f'Total Train Time: {datetime.timedelta(seconds = end_train - start_train)}')
        
    @torch.no_grad()
    def _run_eval(self, epoch, N_steps, N_negs, valid_q1_dict, valid_batch_indices, step=0):
        eval_output = self.eval_epoch(epoch, valid_batch_indices, N_steps, N_negs, valid_q1_dict)
        logger.info(eval_output)
        metric_dict, valid_batch_indices = eval_output['metric_dict'], eval_output['valid_batch_indices']
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
        
        return valid_batch_indices

    @torch.no_grad()
    def eval_epoch(self, epoch, valid_batch_indices, N_steps, N_negs, valid_q1_dict) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        dynamic_valid_data = Dataset(data = self.valid, task=self.args.task, batch_indices=valid_batch_indices, step_size = self.valid_steps)
        self.valid_loader = torch.utils.data.DataLoader(dynamic_valid_data, batch_size=self.args.walks_num*2, shuffle=False, collate_fn=collate, num_workers= self.args.workers, pin_memory=True)
        
        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            batch_dict['only_bfs_embedding'] = True
                
            hr_vector = self.model(**batch_dict)
            sampled_negative_triplets = negative_sampling(self.model, hr_vector[:self.args.walks_num], 
                                                            self.valid_candidates, valid_q1_dict, self.args.valid_walks_num, N_steps, N_negs, self.valid_graph, self.valid_directed_graph)
            
            negative_examples = make_negative_examples(sampled_negative_triplets)
            negative_examples_dataset = Dataset(data=negative_examples, task=self.args.task, negative=True)   
            all_examples = batch_dict['input_batch_data'] + negative_examples_dataset.get_examples()
            examples_dict = collate(all_examples)
            
            batch_size = len(examples_dict['batch_data'])
            
            examples_dict = move_to_cuda(examples_dict)

            for ex in range(batch_size):
                ex_obj = examples_dict['batch_data'][ex]
                head_id = str(ex_obj.head_id)
                relation = str(ex_obj.relation)
                tail_id = str(ex_obj.tail_id)
                k = (head_id, relation, tail_id)
                self.valid_count_check[k] +=1
                
            outputs = self.model(**examples_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=examples_dict)
            outputs = ModelOutput(**outputs)
            logits_hr, labels_hr = outputs.logits_hr, outputs.labels_hr
            loss = self.criterion(logits_hr, labels_hr)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits_hr, labels_hr, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        self.valid_loss.append(round(losses.avg,3))
        
        valid_batch_indices = set()
        valid_count = sorted(self.valid_count_check.items(), key=lambda x: x[1])
        triplets =  [item[0] for item in valid_count[:self.valid_steps*2]]
        for triple in triplets:
            idx = self.valid_triplet_mapping[triple]
            valid_batch_indices.add(random.choice(idx))
        valid_batch_indices = list(valid_batch_indices)[:self.valid_steps]  
        
        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        eval_output =  {'metric_dict': metric_dict, 
                        'valid_batch_indices': valid_batch_indices}
        print("Valid_loss = {}".format(self.valid_loss))
        return eval_output

    def train_epoch(self, epoch, batch_indices, N_steps, N_negs, train_q1_dict):
        shortest_path = []
        
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            self.training_steps,
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
        
        dynamic_train_data = Dataset(data=self.train, task=self.args.task, batch_indices=batch_indices, step_size = self.training_steps)
        self.train_loader = torch.utils.data.DataLoader(dynamic_train_data, batch_size=self.args.walks_num*2, shuffle=False, collate_fn=collate, num_workers= self.args.workers, pin_memory=True)
        
        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            # compute output
            # outputs = results of calling the 'forward' mehod 
            batch_dict['only_bfs_embedding'] = True
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    hr_vector = self.model(**batch_dict)
            else:
                hr_vector = self.model(**batch_dict)
            
            #for checking shortest path 
            head_ids = []
            for ex in range(self.args.walks_num):
                ex_obj = batch_dict['batch_data'][ex]
                head_id = str(ex_obj.head_id)
                head_ids.append(head_id)
            
            
            sampled_negative_triplets, shortest_path_len = negative_sampling(self.model, hr_vector[:self.args.walks_num], self.candidates, train_q1_dict, self.args.walks_num, N_steps, N_negs, self.graph, self.directed_graph, self.G, head_ids)
            shortest_path.append(sum(shortest_path_len) / len(shortest_path_len))      
            negative_examples = make_negative_examples(sampled_negative_triplets)
            negative_examples_dataset = Dataset(data=negative_examples, task=self.args.task, negative=True)
            all_examples = batch_dict['input_batch_data'] + negative_examples_dataset.get_examples()
            examples_dict = collate(all_examples)
            batch_size = len(examples_dict['batch_data'])
            examples_dict = move_to_cuda(examples_dict)
            
            #count entities in batch
            for ex in range(batch_size):
                ex_obj = examples_dict['batch_data'][ex]
                head_id = str(ex_obj.head_id)
                relation = str(ex_obj.relation)
                tail_id = str(ex_obj.tail_id)
                k = (head_id, relation, tail_id)
                self.train_count_check[k] +=1 
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**examples_dict)
            else:
                outputs = self.model(**examples_dict)
            
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=examples_dict)
            outputs = ModelOutput(**outputs)
            logits_hr, labels_hr = outputs.logits_hr, outputs.labels_hr
            logits_tail, logits_rev, labels_rt = outputs.logits_tail, outputs.logits_rev, outputs.labels_rt
            
            
            # head + relation -> tail
            loss = self.criterion(logits_hr, labels_hr)
            # tail -> head + relation
            loss += self.criterion(logits_tail, labels_rt)
            loss += self.criterion(logits_rev, labels_rt)

            acc1, acc3 = accuracy(logits_hr, labels_hr, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
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

        batch_indices = set()
        count = sorted(self.train_count_check.items(), key=lambda x: x[1])
        triples = [item[0] for item in count[:self.training_steps*2]]
        for triple in triples:
            idx = self.train_triplet_mapping[triple]
            batch_indices.add(random.choice(idx))
        batch_indices = list(batch_indices)[:self.training_steps]
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('{} Epoch Training loss: {}'.format(epoch, round(losses.avg,3)))
        self.train_loss.append(round(losses.avg,3))

        avg_path = sum(shortest_path)/ len(shortest_path)
        print(f'average shortest path: {avg_path}')
        self.shortest_pathes.append(avg_path)
        print("Train_loss = {}".format(self.train_loss))
        
        return batch_indices

    def _setup_training(self):
        self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3,4]).cuda('cuda:0')
      

    def _create_lr_scheduler(self, total_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=total_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=total_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
