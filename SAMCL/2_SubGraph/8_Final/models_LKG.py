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

import math
import time
import datetime

def L2_norm(matrix):
    return F.normalize(matrix, p=2, dim=0)
    # It is for the shortest path weight so that the normalized direction 'dim' is the same as the batch direction.

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    inv_b: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.log_inv_b = torch.nn.Parameter(torch.tensor(1.0 / args.B).log(), requires_grad=args.finetune_B)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.idx = 0

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.subgraph = args.subgraph_size * 2   
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
        
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(hr_vector.device)        
        
        if not args.validation:
            center_triple = batch_dict['batch_triple'][0] 
            st_list = self.st_dict[center_triple]
            st_vector = torch.tensor(st_list).reshape(logits.size(0), 1)
            st_vector = torch.sqrt(st_vector)
            st_weight = st_vector.mm(st_vector.t()).to(hr_vector.device)
            st_weight.fill_diagonal_(1)
            st_weight = st_weight.float()
            st_weight *= self.log_inv_b.exp()
            logits += st_weight

        logits *= self.log_inv_t.exp()
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
                'tail_vector': tail_vector.detach()}

 

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
