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

from triplet_mask import construct_mask
from config import args
import pickle

import time
import datetime



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
    inv_tt: torch.tensor
    inv_b: torch.tensor
    Lambda: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    logits_Contra_hr: torch.tensor
    logits_Contra_tail: torch.tensor
    degree_head: torch.tensor
    degree_tail: torch.tensor
    

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.log_inv_tt = torch.nn.Parameter(torch.tensor(1.0 / args.tt).log(), requires_grad=args.finetune_tt)
        self.log_inv_b = torch.nn.Parameter(torch.tensor(1.0 / args.B).log(), requires_grad=args.finetune_B)
        self.Lambda = torch.nn.Parameter(torch.tensor(args.Lambda), requires_grad=args.finetune_Lambda)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.subgraph = args.subgraph_size * 2   
        
        with open(args.shortest_path, 'rb') as fr:
            self.st_dict = pickle.load(fr)    
        
        """
        with open(args.shortest_path_valid, 'rb') as f:
            self.valid_st_dict = pickle.load(f)
        """

        """
        # These lines are for shortest path weight
        self.train_data = json.load(open(args.train_path, 'r', encoding='utf-8'))
        self.valid_data = json.load(open(args.valid_path, 'r', encoding='utf-8'))
        _, _, _, _, self.train_entities = build_graph(args.train_path)
        _, _, _, _, self.valid_entities = build_graph(args.valid_path)
        self.maxlen_train = len(self.train_entities)
        self.maxlen_valid = len(self.valid_entities)
        
        self.degree_dict_train = json.load(open(args.degree_train, 'r', encoding='utf-8'))
        self.degree_dict_valid = json.load(open(args.degree_valid, 'r', encoding='utf-8'))

        self.linkGraph_train = LinkGraph(args.train_path)
        self.linkGraph_valid = LinkGraph(args.valid_path)
        
        self.nxGraph_train = nx.Graph()
        self.nxGraph_valid = nx.Graph()     

        for item in self.train_data:
            self.nxGraph_train.add_node(item["head_id"], label=item["head"])
            self.nxGraph_train.add_node(item["tail_id"], label=item["tail"])
            self.nxGraph_train.add_edge(item["head_id"], item["tail_id"], relation=item["relation"])
        
        for item in self.valid_data:
            self.nxGraph_valid.add_node(item["head_id"], label=item["head"])
            self.nxGraph_valid.add_node(item["tail_id"], label=item["tail"])
            self.nxGraph_valid.add_edge(item["head_id"], item["tail_id"], relation=item["relation"])
        """

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
                aug_hr_token_ids1, aug_hr_token_ids2, aug_tail_token_ids1, aug_tail_token_ids2,
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

        aug_hr_vector1 = self._encode(self.hr_bert,
                                   token_ids=aug_hr_token_ids1,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        aug_tail_vector1 = self._encode(self.tail_bert,
                                   token_ids=aug_tail_token_ids1,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        aug_hr_vector2 = self._encode(self.hr_bert,
                                   token_ids=aug_hr_token_ids2,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        aug_tail_vector2 = self._encode(self.tail_bert,
                                   token_ids=aug_tail_token_ids2,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
       
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector,
                'aug_hr_vector1': aug_hr_vector1,
                'aug_hr_vector2': aug_hr_vector2,
                'aug_tail_vector1': aug_tail_vector1,
                'aug_tail_vector2': aug_tail_vector2}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        aug_hr_vector1, aug_hr_vector2 = output_dict['aug_hr_vector1'], output_dict['aug_hr_vector2']
        aug_tail_vector1, aug_tail_vector2 = output_dict['aug_tail_vector1'], output_dict['aug_tail_vector2']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        logits = hr_vector.mm(tail_vector.t())
        batch_data = batch_dict['batch_triple']
       
        """
        # Shortest Path Weight
        batch_data = batch_dict['batch_triple']
        source = batch_data[0][0]
        st_list = []
        for ex in batch_data:
            target = ex[2]
            if args.validation == False:
                try:
                    st = nx.shortest_path_length(self.nxGraph_train, source=source, target=target)
                    # for head-to-head
                    if st == 0:
                        st = 1
                except nx.NetworkXNoPath:
                    # Disconnected Triples
                    st = self.maxlen_train

            if args.validation == True:        
                try:
                    st = nx.shortest_path_length(self.nxGraph_valid, source=source, target=target)
                    # for head-to-head
                    if st == 0:
                        st = 1
                except nx.NetworkXNoPath:
                    # Disconnected Triples
                    st = self.maxlen_valid
            st_list.append(1/st)
    
        st_value = sum(st_list) / len(st_list)
        
        st_vector = torch.tensor(st_list).view(-1, 1)
        st_vector = st_vector.type(torch.float32)
        st_weight = st_vector.mm(st_vector.t()).to(logits.device)
        st_weight.fill_diagonal_(1)

        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)

        # Case 1. 
        # st_weight = torch.log(L2_norm(st_weight))
        # logits *= self.log_inv_t.exp(

        # Case 2.
        # st_weight = L2_norm(torch.log(st_weight) * self.log_inv_t) 
        # logits *= st_weight

        # Case 3. ST Weight with Learnable parameter b
        """

        logits *= self.log_inv_t.exp()
        
        if not args.validation:
            st_vector = torch.zeros(logits.size(0), 1).to(logits.device)
            source = batch_data[0][0]
            for i, triple in enumerate(batch_data):
                target = triple[2]
                st = 1 / self.st_dict[source][target]
                st_vector[i][0] = st
            st_weight = st_vector.mm(st_vector.t()).to(logits.device)
            st_weight.fill_diagonal_(1)
            
            # Leave the original scoring function alone
            # The multiplication between ST weight and the scoring function induces the large variation of the loss
            # The large variation ratio is not good for deep learning model because it can cause the overfitting               
            # The ST weight is only for training because the validaiton Graph have too many disconnected Graph.
            st_margin = st_weight * self.log_inv_b.exp()
            logits += st_margin

        # """
        degree_head = torch.zeros(logits.size(0)).to(logits.device)
        degree_tail = torch.zeros(logits.size(0)).to(logits.device)

        for i, triple in enumerate(batch_data):
            head, tail = triple[0], triple[2]
            if not args.validation:
                dh = len(self.linkgraph_train.get_neighbor_ids(head, 1)) + 1
                dt = len(self.linkgraph_train.get_neighbor_ids(tail, 1)) + 1
            if args.validation:
                dh = len(self.linkgraph_valid.get_neighbor_ids(head, 1)) + 1
                dt = len(self.linkgraph_valid.get_neighbor_ids(tail, 1)) + 1
            degree_head[i] = dh
            degree_tail[i] = dt
        degree_head, degree_tail = degree_head.log(), degree_tail.log()

        # ContraREC Loss
        logits_Contra_hr = aug_hr_vector1.mm(aug_hr_vector2.t()).to(logits.device)
        logits_Contra_tail = aug_tail_vector2.mm(aug_tail_vector2.t()).to(logits.devcie)

        logits_Contra_hr *= self.log_inv_tt
        logits_Contra_tail *= self.log_inv_tt

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)
        

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
        
        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        
        return {'logits': logits,
                'logits_Contra_hr': logits_Contra_hr,
                'logits_Contra_tail': logits_Contra_tail,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'inv_b': self.log_inv_b.detach().exp(),
                'inv_tt': self.log_inv_tt.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach(),
                'self.Lambda': self.Lambda.detach(),
                'degree_head': degree_head,
                'degree_tail': degree_tail}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

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
