from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from triplet_mask import construct_mask_hr, construct_mask_tail

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits_hr: torch.tensor
    logits_tail: torch.tensor
    logits_reverse: torch.tensor
    labels_hr: torch.tensor
    labels_tail: torch.tensor
    inv_t: torch.tensor
    inv_t_hard: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.log_inv_t_hard = torch.nn.Parameter(torch.tensor(1.0 / args.tt).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.negative_size = args.negative_size * 2

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
            
        self.labels_hr = torch.arange(0,int(self.batch_size // 2), int(self.negative_size//2)).long()
        self.labels_tail = torch.arange(self.batch_size // self.negative_size)
        

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
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)

        # labels_hr = torch.zeros(batch_size // self.negative_size, dtype = torch.long).to(hr_vector.device)
        labels_hr = self.labels_hr.to(hr_vector.device)
        labels_tail = self.labels_tail.to(hr_vector.device)

        ## logits_hr
        hr1 = hr_vector[::self.negative_size] # all Postive hr, (batch_size, 768)
        tail1 = tail_vector[::2] ## all positive tail 
        logits_hr = torch.mm(hr1, tail1.t())

        ## logits_tail
        tail2 = tail_vector[::self.negative_size]
        logits_tail = torch.mm(tail2, hr1.t())
        
        ## logits_reverse
        hr_reverse = hr_vector[1::self.negative_size]
        tail_reverse = tail_vector[1::self.negative_size]
        logits_reverse = torch.mm(hr_reverse, tail_reverse.t())
 
        if self.training:
            margin = torch.zeros(logits_hr.size()).to(logits_hr.device)
            margin[:,0] = 1 * self.add_margin
            
            logits_hr -= margin.to(logits_hr.device)
            logits_tail -= torch.zeros(logits_tail.size()).fill_diagonal_(self.add_margin).to(logits_hr.device)
            logits_reverse -= torch.zeros(logits_reverse.size()).fill_diagonal_(self.add_margin).to(logits_hr.device)
        
        ## for hard negative -> inv * 2
        hard = torch.ones(self.batch_size // self.negative_size, self.batch_size //2).to(logits_hr.device)
        hard_inv = (torch.arange(hard.size(1)) // (self.negative_size//2) == torch.arange(len(hard)).unsqueeze(1)).to(logits_hr.device)
        hard[hard_inv] *= self.log_inv_t_hard.exp()
        hard[hard == 1] *= self.log_inv_t.exp()
       
        logits_hr *= hard
        logits_tail *= self.log_inv_t.exp()
        logits_reverse *= self.log_inv_t.exp()


        """
        # logits_reverse's mask is equal to the triplet_mask_tail
        """

        triplet_mask_hr = construct_mask_hr(row_exs = batch_dict['batch_data']).to(logits_hr.device)
        triplet_mask_tail = construct_mask_tail(row_exs = batch_dict['batch_data']).to(logits_hr.device)
        if triplet_mask_hr is not None:
            logits_hr.masked_fill_(~triplet_mask_hr, -1e4)
            logits_tail.masked_fill_(~triplet_mask_tail, -1e4)
            logits_reverse.masked_fill_(~triplet_mask_tail, -1e4)
        
        

        return {'logits_hr': logits_hr,
                'logits_tail': logits_tail,
                'logits_reverse': logits_reverse,
                'labels_hr': labels_hr,
                'labels_tail': labels_tail,
                'inv_t': self.log_inv_t.detach().exp(),
                'inv_t_hard': self.log_inv_t.detach().exp(),
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
