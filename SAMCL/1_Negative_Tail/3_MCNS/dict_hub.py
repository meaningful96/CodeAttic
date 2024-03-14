import os
import glob
import json

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger
from typing import Optional, List
from collections import defaultdict

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer


def count_dict(path, examples: List[list], path_length):
    triplets  = json.load(open(path, 'r', encoding='utf-8'))
    count_check = {}
    for i in range(len(triplets)):
        head_id = triplets[i]['head_id']
        tail_id = triplets[i]['tail_id']
        relation = triplets[i]['relation']
        inv_relation = 'inverse '+ relation
        k = (head_id, relation, tail_id)
        inv_k = (tail_id, inv_relation, head_id)
        count_check[k] = 0
        count_check[inv_k] = 0
        
    cnt = len(examples)
    triplet_mapping = defaultdict(list)
    
    for i in range(cnt):
        for j in range(path_length):
            head_id = examples[i][j]['head_id']
            tail_id = examples[i][j]['tail_id']
            relation = examples[i][j]['relation']
            inv_relation = 'inverse '+ relation
            k = (head_id, relation, tail_id)
            inv_k = (tail_id, inv_relation, head_id)
        
            triplet_mapping[k].append(i)
            triplet_mapping[inv_k].append(i)
      
    return count_check, triplet_mapping
