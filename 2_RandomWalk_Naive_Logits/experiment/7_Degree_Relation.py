"""
Created on meaningful96

DL Project
"""

import json
from typing import List, Dict, Tuple
from collections import defaultdict, deque
import random
import time
import datetime
import numpy as np

class LinkGraph:
    def __init__(self, train_path: str):
        print('Start to build link graph!!!')
        # id -> {(relation, id)}
        self.graph = defaultdict(set)
        
        # Directed Graph
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id,tail_id = ex['head_id'], ex['tail_id']
            
            # Add to graph with all details
            if head_id not in self.graph:
                self.graph[head_id] = set()
            # Link Graph(Undirected Graph)
            self.graph[head_id].add(tail_id) 

            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)           
        
        print('Done building link graph with {} nodes'.format(len(self.graph)))

    def counting_hops(self):
        hop_dictionary = {f"{i} hop": 0 for i in range(1, 101)}
        head_list = list(self.graph.keys())
        over_100 = 0
        under_100 = 0
        for head in head_list:
            if len(self.graph[head]) <= 100:
                hop_dictionary['{} hop'.format(len(self.graph[head]))] += 1
                under_100 += 1
            else:
                over_100 += 1
        print()
        print("Total Entity: {}".format(len(head_list)))
        print("Under Degree 100: {}".format(under_100))
        print()
        return hop_dictionary, over_100
         
# Load data    
train_path_wn = "/home/youminkk/Model_Experiment/3_Random_Walk_epoch/data/WN18RR/train.txt.json"
train_path_fb = "/home/youminkk/Model_Experiment/3_Random_Walk_epoch/data/FB15k237/train.txt.json"
    

#%%
data_wn = json.load(open(train_path_wn, 'r', encoding='utf-8'))

# Set the relation
relation_counts = defaultdict(int)
relation_id_mapping = {}
current_id = 1

for ex in data_wn:
    relation = ex['relation']
    if relation not in relation_id_mapping:
        relation_id_mapping[relation] = current_id
        current_id += 1
    relation_counts[relation_id_mapping[relation]] += 1

# Print the mapping of relation names to ID numbers
print("Relation Name to ID Mapping:")
for relation, relation_id in relation_id_mapping.items():
    print(f"{relation}: {relation_id}")

# Print the counts for each relation ID
print("\nRelation ID Counts:")
for relation_id, count in relation_counts.items():
    print(f"Relation ID: {relation_id}, Count: {count}")

data_fb = json.load(open(train_path_fb, 'r', encoding='utf-8'))

# Set the relation
relation_counts = defaultdict(int)
relation_id_mapping = {}
current_id = 1

for ex in data_fb:
    relation = ex['relation']
    if relation not in relation_id_mapping:
        relation_id_mapping[relation] = current_id
        current_id += 1
    relation_counts[relation_id_mapping[relation]] += 1

# Print the mapping of relation names to ID numbers
print("Relation Name to ID Mapping:")
for relation, relation_id in relation_id_mapping.items():
    print(f"{relation}: {relation_id}")

# Print the counts for each relation ID
print("\nRelation ID Counts:")
for relation_id, count in relation_counts.items():
    print(f"Relation ID: {relation_id}, Count: {count}")
    
