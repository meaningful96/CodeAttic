from bfs import *
from logger_config import logger
from collections import defaultdict

import torch
import pickle
import time
import datetime

def load_pkl(path):
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

path = '/home/youminkk/Model_Experiment/2_SubGraph/7_BFS/data/FB15k237/train_bfs.pkl'
batch_size = 3072
subgraph = 1536
step_size = 177
Epoch = 20

train_path = '/home/youminkk/Model_Experiment/2_SubGraph/7_BFS/data/FB15k237/train.txt.json'
obj = BFS(train_path)


Graph1, Graph_tail1, diGraph, appearance1, entities1 = build_graph(train_path)
Graph2, Graph_tail2, diGraph2, appearance2, entities2 = build_graph(train_path)
Graph3, Graph_tail3, diGraph3, appearance3, entities3 = build_graph(train_path)   
Graph4, Graph_tail4, diGraph4, appearance4, entities4 = build_graph(train_path)
Graph5, Graph_tail5, diGraph5, appearance5, entities5 = build_graph(train_path) 
logger.info("Building Graph Done!!")

data_dict = load_pkl(path)
num_candidates = step_size * batch_size // (subgraph * 2)
initial_triples = random.sample(list(data_dict.keys()), num_candidates)


batch_duple = []
s_t = time.time()
logger.info('Start!!!!!')

logger.info("Stage1 !!")
for epoch in range(Epoch):
    if epoch == 0:
        cand = initial_triples
        train_data, counted_appearance1, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance1, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train1 = sorted(counted_appearance1.items(), key=lambda x: x[1])
        new_candidates_train1 = sorted_candidates_train1[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train1]
        appearance1 = counted_appearance1

    else:
        train_data, counted_appearance1, sub_total, batch_total = Making_Subgraph(diGraph,data_dict, cand, subgraph, appearance1, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train1 = sorted(counted_appearance1.items(), key=lambda x: x[1])
        new_candidates_train1 = sorted_candidates_train1[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train1]
        appearance1 = counted_appearance1
logger.info("Stage2 !!")
for epoch in range(Epoch):
    if epoch == 0:
        cand = initial_triples
        train_data, counted_appearance2, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance2, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train2 = sorted(counted_appearance2.items(), key=lambda x: x[1])
        new_candidates_train2 = sorted_candidates_train2[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train2]
        appearance2 = counted_appearance2

    else:
        train_data, counted_appearance2, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance2, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train2 = sorted(counted_appearance2.items(), key=lambda x: x[1])
        new_candidates_train2 = sorted_candidates_train2[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train2]
        appearance2 = counted_appearance2
logger.info("Stage3 !!")
for epoch in range(Epoch):
    if epoch == 0:
        cand = initial_triples
        train_data, counted_appearance3, sub_total, batch_total = Making_Subgraph(diGraph,data_dict, cand, subgraph, appearance3, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train3 = sorted(counted_appearance3.items(), key=lambda x: x[1])
        new_candidates_train3 = sorted_candidates_train3[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train3]
        appearance3 = counted_appearance3

    else:
        train_data, counted_appearance3, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance3, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train3 = sorted(counted_appearance3.items(), key=lambda x: x[1])
        new_candidates_train3 = sorted_candidates_train3[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train3]
        appearance3 = counted_appearance3

logger.info("Stage 4!!")
for epoch in range(Epoch):
    if epoch == 0:
        cand = initial_triples
        train_data, counted_appearance4, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance4, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train4 = sorted(counted_appearance4.items(), key=lambda x: x[1])
        new_candidates_train4 = sorted_candidates_train4[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train4]
        appearance4 = counted_appearance4

    else:
        train_data, counted_appearance4, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance4, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train4 = sorted(counted_appearance4.items(), key=lambda x: x[1])
        new_candidates_train4 = sorted_candidates_train4[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train4]
        appearance4 = counted_appearance4
        
logger.info("Stage 5!!")
for epoch in range(Epoch):
    if epoch == 0:
        cand = initial_triples
        train_data, counted_appearance5, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance5, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train5 = sorted(counted_appearance5.items(), key=lambda x: x[1])
        new_candidates_train5 = sorted_candidates_train5[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train5]
        appearance5 = counted_appearance5

    else:
        train_data, counted_appearance5, sub_total, batch_total = Making_Subgraph(diGraph, data_dict, cand, subgraph, appearance5, batch_size)
        batch_duple.extend(batch_total)

        sorted_candidates_train5 = sorted(counted_appearance5.items(), key=lambda x: x[1])
        new_candidates_train5 = sorted_candidates_train5[:num_candidates]
        cand = [item[0][0] for item in new_candidates_train5]
        appearance5 = counted_appearance5


e = time.time()

dicts = [appearance1, appearance2, appearance3, appearance4, appearance5]  # List of dictionaries
combined_dict = {k: int(sum(d.get(k, 0) for d in dicts) / len(dicts)) for d in dicts for k in d}

logger.info("Done!!")
output_path = '/home/youminkk/fb_bfs.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(combined_dict, f)
print(combined_dict)
logger.info("Saving Done!!!")
