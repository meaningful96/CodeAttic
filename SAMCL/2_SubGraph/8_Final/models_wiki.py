from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
import networkx as nx
import json
import torch.nn.functional as F

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from randomwalk import build_graph
from typing import List, Dict
from collections import defaultdict

from triplet_mask import construct_mask
from config import args
import pickle

import time
import datetime

def L2_norm(matrix):
    return F.normalize(matrix, p=2, dim=0)
    # It is for the shortest path weight so that the normalized direction 'dim' is the same as the batch direction.

def build_model(args) -> nn.Module:
    return CustomBertModel(args)

def linkGraph(train_path:str):
    Graph, Graph_tail, diGraph = defaultdict(set), defaultdict(set), defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding = 'utf-8'))
    appearance = {}

    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0

        if head_id not in Graph:
            Graph[head_id] = set()
        Graph[head_id].add((head_id, relation, tail_id))
        
        if tail_id not in Graph:
            Graph[tail_id] = set()
        Graph[tail_id].add((tail_id, relation, head_id))    

    entities = list(Graph.keys())

    return Graph, entities

class LinkGraph:
    def __init__(self, train_path:str):
        self.Graph, _ = linkGraph(train_path)

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[int]: # mapping: tail_id:str, List[str] -> tail_id:int, List[int]
        if n_hops <= 0:
            return []

        neighbors = [item[2]for item in self.Graph.get(tail_id, set())]
        distant_neighbors = []
        
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))
        return list(set(neighbors + distant_neighbors)) 

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    inv_b: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    degree_head: torch.tensor
    degree_tail: torch.tensor

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.log_inv_b = torch.nn.Parameter(torch.tensor(1.0 / args.B).log(), requires_grad=args.finetune_B)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.offset = 0

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.subgraph = args.subgraph_size * 2   
        
        self.degree_train = json.load(open(args.degree_train, 'r', encoding='utf-8'))
        self.degree_valid = json.load(open(args.degree_valid, 'r', encoding='utf-8'))
        self.linkGraph_train = LinkGraph(args.train_path)
        self.linkGraph_valid = LinkGraph(args.valid_path)

        self.tail_bert = deepcopy(self.hr_bert).to("cuda:2") ##
        with open(args.shortest_path, 'rb') as file:
            self.st_dict = pickle.load(file)
          

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        logits = hr_vector.mm(tail_vector.t())
        batch_data = batch_dict['batch_triple']
        
        # Ver1. Logits + Shortest Weight Matrix
        # Case 3. ST Weight with Learnable parameter b  
         
        st_list = self.st_dict[batch_data[0]]
        st_vector = torch.tensor(st_list).reshape(logits.size(0), 1)
        st_weight = st_vector.mm(st_vector.t()).to(hr_vector.device)
        st_weight.fill_diagonal_(1)
        st_weight *= self.log_inv_b.exp()

        logits += st_weight
        logits *= self.log_inv_t.exp()

        del st_vector
        del st_weight

        if not args.validation:
            dh = self.degree_train[batch_data[0]]['dh']    
            dt = self.degree_train[batch_data[0]]['dt']
            dh, dt = torch.tensor(dh), torch.tensor(dt)
        if args.validation:
            dh = self.degree_valid[batch_data[0]]['dh']    
            dt = self.degree_valid[batch_data[0]]['dt']
            dh, dt = torch.tensor(dh), torch.tensor(dt)
        degree_head = degree_head.log()
        degree_tail = degree_tail.log()

        triplet_mask = batch_dict.get('triplet_mask', None).to(hr_vector.device)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)        

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = (torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()).to(hr_vector.device)
            self_negative_mask = batch_dict['self_negative_mask'].to(hr_vector.device)
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        
        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'inv_b': self.log_inv_b.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach(),
                'degree_head': degree_head,
                'degree_tail': degree_tail}

 

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
