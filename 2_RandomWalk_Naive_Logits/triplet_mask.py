import torch

from typing import List

from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict

entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


def construct_mask_hr(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_zeros = col_exs is None
    col_exs = row_exs if col_exs is None else col_exs

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if positive_on_zeros else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])

    negative_size = args.negative_size
    row_entity_ids = row_entity_ids[::2]
    col_entity_ids = col_entity_ids[::2]
    num_row, num_col = len(row_entity_ids), len(col_entity_ids)
    # num_row x num_col
    row_exs = row_exs[::(2*negative_size)]
    col_exs = col_exs[::2]

    triplet_mask = torch.full((len(row_exs), num_col), True, dtype=bool)
    num_row = len(row_exs)

    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if len(neighbor_ids) <= 1:
            continue
        for j in range(num_col):
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False
            if i * negative_size == j and positive_on_zeros:
                triplet_mask[i][j] = True
    
    return triplet_mask


def construct_mask_tail(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    col_exs = row_exs if col_exs is None else col_exs

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    # num_row x num_col
    negative_size = args.negative_size 
    row_entity_ids = row_entity_ids[::2]
    col_entity_ids = col_entity_ids[::2]
    row_entity_ids = row_entity_ids[::negative_size]
    col_entity_ids = col_entity_ids[::negative_size]
    num_row = len(row_entity_ids)
    num_col = len(col_entity_ids)

    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_mask_valid(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_self_negative_mask(exs: List) -> torch.tensor:
    negative_size = args.negative_size*2   
    example = exs[::negative_size]
    mask = torch.ones(len(example))
    for idx, ex in enumerate(example):
        head_id, relation = ex.head_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if head_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()
