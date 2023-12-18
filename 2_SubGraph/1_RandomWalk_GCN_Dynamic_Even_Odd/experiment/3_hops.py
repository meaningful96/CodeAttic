#%%

import json
from typing import List, Dict, Tuple
from collections import defaultdict, deque
import random
import time
import datetime
import numpy as np


# Load data    
train_path_fb = "/home/youminkk/Paper_reconstruction/2_SimKGC_HardNegative/data/WN18RR/train_origin.txt.json"
with open(train_path_fb, 'r', encoding='utf-8') as f:
    train_data_fb = json.load(f)

path = "/home/youminkk/Paper_reconstruction/3_SimKGC_HN_with_randomwalk/data/WN18RR/random_walk_train_512.json"
with open(path, 'r', encoding='utf-8') as f:
    data_fb = json.load(f)

import networkx as nx
            
G_fb = nx.Graph()
for item in train_data_fb:
    head_id = item['head_id']
    tail_id = item['tail_id']
    relation = item['relation']
    
    G_fb.add_node(head_id, name=item['head'])      
    G_fb.add_node(tail_id, name=item['tail'])
    G_fb.add_edge(head_id, tail_id, relation=relation)  


st_for_samples = []
for ex in data_fb:
    tmp = []
    for j in range(len(ex)):
        try:
            shortest_path = nx.shortest_path_length(G_fb, source = ex[0]['head_id'], target = ex[j]['head_id'])
            tmp.append(shortest_path)
        except nx.NetworkXNoPath:
            tmp.append(0)
    st_for_samples.append(tmp)
    

fb_hops = {'1 hop': 0, '2 hop': 0, '3 hop': 0, '4 hop': 0, '5 hop': 0,
           '6 hop': 0, '7 hop': 0, '8 hop': 0, '9 hop': 0, '10 hop': 0,
           '11 hop': 0, '12 hop': 0, '13 hop': 0, '14 hop': 0, '15 hop': 0}


for path_lengths in st_for_samples:
    for length in path_lengths:
        # Check if the hop count is within manageable range
        if 1 <= length <= 15:
            key = f"{length} hop"
            fb_hops[key] += 1
fb_hops = {key: value/len(st_for_samples) for key, value in fb_hops.items()}       
print(fb_hops)
