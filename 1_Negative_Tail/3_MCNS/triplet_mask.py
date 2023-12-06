import torch

from typing import List

from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict

entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


def construct_mask_tail(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    col_exs = row_exs if col_exs is None else col_exs

    pos = args.walks_num

    row_pos, row_pos_rev = row_exs[:pos], row_exs[pos:(2*pos)]
    roe_neg, row_neg_rev = row_exs[(2*pos)::2], row_exs[(2*pos+1)::2]

    col_pos, col_pos_rev = col_exs[:pos], col_exs[pos:(2*pos)]
    col_neg, col_neg_rev = col_exs[(2*pos)::2], col_exs[(2*pos+1)::2]


    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_pos])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_pos])
   
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)


    num_row, num_col = len(row_pos), len(col_pos)
    # mask out other possible neighbors
    for i in range(num_col):
        head_id, relation = col_pos[i].head_id, col_pos[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_row):
            if i == j and positive_on_diagonal:
                continue
            tail_id = row_pos[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_mask_rev(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    col_exs = row_exs if col_exs is None else col_exs

    pos = args.walks_num

    row_pos, row_pos_rev = row_exs[:pos], row_exs[pos:(2*pos)]
    roe_neg, row_neg_rev = row_exs[(2*pos)::2], row_exs[(2*pos+1)::2]

    col_pos, col_pos_rev = col_exs[:pos], col_exs[pos:(2*pos)]
    col_neg, col_neg_rev = col_exs[(2*pos)::2], col_exs[(2*pos+1)::2]


    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_pos_rev])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_pos_rev])
   
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    num_row, num_col = len(row_pos_rev), len(col_pos_rev)
    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_pos_rev[i].head_id, row_pos_rev[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_pos_rev[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_mask_hr(row_exs: List, col_exs: List = None) -> torch.tensor:
    col_exs = row_exs if col_exs is None else col_exs

    pos = args.walks_num

    row_pos, row_pos_rev = row_exs[:pos], row_exs[pos:(2*pos)]
    roe_neg, row_neg_rev = row_exs[(2*pos)::2], row_exs[(2*pos+1)::2]

    col_pos, col_pos_rev = col_exs[:pos], col_exs[pos:(2*pos)]
    col_neg, col_neg_rev = col_exs[(2*pos)::2], col_exs[(2*pos+1)::2]

    tail = col_pos + col_neg

    # num_row x num_col
    triplet_mask = torch.full((len(row_pos), len(tail)), True, dtype=bool)

    num_row, num_col = len(row_pos), len(tail)
    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_pos[i].head_id, row_pos[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j:
                continue
            tail_id = tail[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_self_negative_mask(exs: List) -> torch.tensor:
    exs = exs[:args.walks_num]
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        head_id, relation = ex.head_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if head_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()
