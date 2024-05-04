from collections import defaultdict
from randomwalk import Biased_RandomWalk, Making_Subgraph 
import random
import networkx as nx
import time
import datetime
import json
import pickle

def build_graph(train_path:str):
    Graph, Graph_tail, diGraph = defaultdict(set), defaultdict(set), defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding = 'utf-8'))
    appearance = {}

    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0

        if head_id not in Graph:
            Graph[head_id] = set()
            Graph_tail[head_id] = set()
            diGraph[head_id] = set()
        Graph[head_id].add((head_id, relation, tail_id))
        Graph_tail[head_id].add(tail_id)
        diGraph[head_id].add((head_id, relation, tail_id))
        
        if tail_id not in Graph:
            Graph[tail_id] = set()
            Graph_tail[tail_id] = set()
        Graph[tail_id].add((tail_id, relation, head_id))
        Graph_tail[tail_id].add(head_id)    

    entities = list(Graph.keys())

    return Graph, Graph_tail, diGraph, appearance, entities


def counting_loop(epochs, data_dict, initial_triples, subgraph_size, batch_size, appearance, num_candidates):
    
    epoch_duplication_avg = []
    avg = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        start_epoch = time.time()
        
        if epoch == 0:
            candidates = initial_triples
        
        data, counted_appearance, sub_total, batch_total = Making_Subgraph(data_dict,
                                                                           candidates, 
                                                                           subgraph_size, 
                                                                           appearance, 
                                                                           batch_size)
        
        batch_duplication = sum(batch_total) / len(batch_total)
        sub_duplication = sum(sub_total) / len(sub_total)
        epoch_duplication_avg.append((batch_duplication, sub_duplication))
        avg.append(batch_duplication)
        sorted_candidates = sorted(counted_appearance.items(), key=lambda x: x[1])
        new_candidates = sorted_candidates[:num_candidates]
        candidates = [item[0] for item in new_candidates]
        
        appearance = counted_appearance
        
        end_epoch = time.time()
        print(f"Epoch {epoch+1} Time = '{datetime.timedelta(seconds = end_epoch - start_epoch)}'")

    print("Epoch Average Duplication:")
      
    # Counting the triples that were never used
    zero_count = sum(value == 0 for value in appearance.values())
    print(f"Never Used Triplets: {zero_count}")
    print(f"Batch Duple: {sum(avg) / len(avg)}")
if __name__ == "__main__":
    path = './data/FB15k237/train_antithetical_40_300.pkl'
    train_path = './data/FB15k237/train.txt.json'    
    train_data = json.load(open(train_path, 'r', encoding='utf-8'))

    Graph, Graph_tail, diGraph, appearance, entities = build_graph(train_path)
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)

    subgraph_size = 900
    batch_size = subgraph_size*2
    num_candidates = len(train_data)//batch_size

    epochs = 20
    initial_triples = random.sample(list(data_dict.keys()), num_candidates)

    counting_loop(epochs, data_dict, initial_triples, subgraph_size, batch_size, appearance, num_candidates)
    print(f"The Path: {train_path}")
