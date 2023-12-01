from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask_hr, construct_mask_tail, construct_mask_rev, construct_self_negative_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits_hr: torch.tensor
    logits_tail: torch.tensor
    logits_rev: torch.tensor
    labels_hr: torch.tensor
    labels_rt: torch.tensor
    inv_t: torch.tensor
    inv_tt: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.log_inv_tt = torch.nn.Parameter(torch.tensor(1.0 / args.tt).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        #self.batch_size = args.batch_size
        self.offset = 0
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.hard = torch.ones(args.walks_num, (args.N_negs + 1)*args.walks_num)
        self.hard_inv = torch.full(self.hard.size(), True, dtype = bool)
        self.walks_num = args.walks_num    

        tmp = args.walks_num
        for i in range(self.hard.size(0)):
            for j in range(self.hard.size(1)):
                if j == args.walks_num:
                    for p in range(args.N_negs):
                        self.hard_inv[i][tmp + p] = False
                    tmp += args.N_negs
       


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
                only_ent_embedding=False, only_bfs_embedding = False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        if only_bfs_embedding:
            hr_vector = self._encode(self.hr_bert,
                                token_ids=hr_token_ids,
                                mask=hr_mask,
                                token_type_ids=hr_token_type_ids)
            return hr_vector
            

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        
        pos = self.walks_num

        hr_pos, hr_pos_rev = hr_vector[:pos], hr_vector[pos:(pos*2)]
        hr_neg, hr_neg_rev = hr_vector[(pos*2)::2], hr_vector[(pos*2+1)::2]

        tail_pos, tail_pos_rev = tail_vector[:pos], tail_vector[pos:(pos*2)]
        tail_neg, tail_neg_rev = tail_vector[(pos*2)::2], tail_vector[(pos*2+1)::2]

        # logits_hr
        hr1, tail1 = hr_pos, torch.cat((tail_pos, tail_neg), dim=0)
        logits_hr = torch.mm(hr1, tail1.t()).to(hr_vector.device)
        labels_hr = torch.arange(logits_hr.size(0)).to(hr_vector.device)

        # logits_tail
        logits_tail = torch.mm(tail_pos, hr_pos.t()).to(hr_vector.device)
        labels_rt = torch.arange(logits_tail.size(0)).to(hr_vector.device)

        # logits_rev
        logits_rev = torch.mm(hr_pos_rev, tail_pos_rev.t()).to(hr_vector.device)

        if self.training:
            margin = torch.zeros(logits_hr.size()).to(hr_vector.device)
            for i in range(margin.size(0)):
                for j in range(margin.size(1)):
                    if i == j:
                        margin[i][j] == self.add_margin
            logits_hr -= margin.to(hr_vector.device)
            logits_tail -= torch.zeros(logits_tail.size()).fill_diagonal_(self.add_margin).to(hr_vector.device)
            logits_rev -= torch.zeros(logits_rev.size()).fill_diagonal_(self.add_margin).to(hr_vector.device)

        hard, hard_inv = self.hard.to(hr_vector.device), self.hard_inv.to(hr_vector.device)
        hard[hard_inv] *= self.log_inv_t.exp()
        hard[hard == 1] *= self.log_inv_tt.exp()

        logits_hr *= hard
        logits_tail *= self.log_inv_t.exp()
        logits_rev *= self.log_inv_t.exp()

        triplet_mask_hr = construct_mask_hr(row_exs = batch_dict['batch_data']).to(hr_vector.device)
        triplet_mask_tail = construct_mask_tail(row_exs = batch_dict['batch_data']).to(hr_vector.device)
        triplet_mask_rev = construct_mask_rev(row_exs = batch_dict['batch_data']).to(hr_vector.device)
        if triplet_mask_hr is not None:
            logits_hr.masked_fill_(~triplet_mask_hr, -1e4)
            logits_tail.masked_fill_(~triplet_mask_tail, -1e4)
            logits_rev.masked_fill_(~triplet_mask_rev, -1e4)

        # self negative
        self_negative_logits = (torch.sum(hr_pos * hr_pos_rev, dim=1) *self.log_inv_tt.exp()).to(logits_hr.device)
        print(self_negative_logits.size())
        self_negative_mask = construct_self_negative_mask(batch_dict['batch_data']).to(logits_hr.device)
        print(self.negative_mask.size())
        self_negative_logits.masked_fill_(~self_negative_mask, -1e4)
        logits_hr = torch.cat([logits_hr, self_negative_logits.unsqueeze(1)], dim=-1)

        return {'logits_hr': logits_hr,
                'logits_tail': logits_tail,
                'logits_rev': logits_rev,
                'labels_hr': labels_hr,
                'labels_rt': labels_rt, 
                'inv_t': self.log_inv_t.detach().exp(),
                'inv_tt': self.log_inv_tt.detach().exp(),
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
