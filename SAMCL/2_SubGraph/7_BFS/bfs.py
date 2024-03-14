from collections import defaultdict, deque
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
from logger_config import logger

import multiprocessing
import datetime
import random
import time
import json
import pickle

def worker(entity, bfs_instance):
    return entity, bfs_instance.bfs_list(entity)

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

def weighted_random_selection(prob_distribution):
    # Generating a list of indices based on the probability distribution
    indices = list(range(len(prob_distribution)))
    
    # Performing weighted random selection for a single trial
    selected_index = random.choices(indices, weights=prob_distribution, k=1)[0]

    return selected_index


class BFS:
    def __init__(self, train_path:str):
        self.Graph, self.Graph_tail, self.diGraph, self.appearance, self.entities = build_graph(train_path)
        logger.info("Done building Link Graph with {} nodes".format(len(self.Graph)))
        logger.info("Done building Directed Graph with {} nodes".format(len(self.diGraph)))
        self.train_dataset = json.load(open(train_path, 'r', encoding='utf-8'))

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[int]: # mapping: tail_id:str, List[str] -> tail_id:int, List[int]
        if n_hops <= 0:
            return []

        neighbors = [item[2] for item in self.Graph.get(tail_id, set())]
        distant_neighbors = []
        
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))
        return list(set(neighbors + distant_neighbors))

    def get_neighbor_ent_ids(self, entity_id: int) -> List[int]:
        neighbor_ids = self.Graph_tail.get(entity_id, set())
        return sorted(list(neighbor_ids))

    def bfs(self, start: int) -> Dict[str, int]:
        visited = {}
        queue = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()
            if node not in visited:
                visited[node] = depth
                for neighbor in self.Graph.get(node, []):  # Link graph
                    queue.append((neighbor[2], depth + 1))

        return visited

    def Departing(self):
        graph = self.Graph  
        all_entities = list(self.Graph.keys())
        entities = list(self.diGraph.keys())
        candidates = list(set(all_entities) - set(entities))
        center_triples = list(random.sample(entities, 10))
        
        tmp = []
        for ex in center_triples:
            length = len(self.bfs(ex))
            tmp.append(length)
        center_length = max(tmp)

        fully_disconnected = []
        for entity in candidates:
            bfs = self.bfs(entity)  
            if len(bfs) != center_length:
                fully_disconnected.append(entity)
        disconnected_triple = []
        for entity in fully_disconnected:
            if entity in self.diGraph.keys():
                disconnected_triple.extend(list(self.diGraph[entity]))
            if entity not in self.diGraph.keys():
                if entity in self.Graph.keys():
                    cand = list(self.Graph[entity])
                    disconnected_triple.extend(rearrange_list(cand))

        return fully_disconnected, disconnected_triple

    def bfs_list(self, entity:str) -> List[list]:
        bfs_dict = self.bfs(entity)
        bfs_list = list(bfs_dict.keys())
        return bfs_list

    def parallel_bfs_lists(self, num_processes: int = None) -> Dict[str, List[str]]:
        entities = self.entities
        
        # Initialize a pool of workers
        with Pool(processes=num_processes) as pool:
            # Create a tuple with the entity and a reference to the bfs instance for each entity
            tasks = [(entity, self) for entity in entities]
            # Use starmap to pass the tuple to the worker function
            results = pool.starmap(worker, tasks)

        # Convert the list of tuples to a dictionary
        bfs_lists_dict = dict(results)
        return bfs_lists_dict

def BFS_Dictionary(obj, num_processes: int):
    dictionary =  obj.parallel_bfs_lists(num_processes)
    return dictionary

def Making_Subgraph(diGraph, bfs_dict, candidates, subgraph_size, appearance, batch_size):
    total_subgraph = []
    batch_total, sub_total = [], []
    tmp1, tmp2 = [], []
    p = len(candidates)
    for candidate in candidates:
        subgraph = []
        entity_list = bfs_dict[candidate]

        # Using Set()
        hr_set = set()
        
        for entity in entity_list:
            triples = diGraph[entity]
            random.shuffle(list(triples))
            for triple in triples:
                h, t = triple[0], triple[2]    
                if h not in hr_set and t not in hr_set:
                    subgraph.append(triple)
                    hr_set.add(h)
                    hr_set.add(t)

        if len(subgraph) < subgraph_size:
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1])
            num_diff = subgraph_size - len(subgraph)
            new_triples = sorted_triples[p: p + num_diff]
            triples = [item[0] for item in new_triples]
            p += num_diff
            subgraph.extend(triples)
            if p + subgraph_size >= len(appearance.keys()):
                p = 0

        if len(subgraph) >= subgraph_size:
            subgraph = subgraph[:subgraph_size]

        assert len(subgraph) == subgraph_size
        
        total_subgraph.extend(subgraph)
        
        for ex in subgraph:
            tmp1.append(ex[0])
            tmp1.append(ex[2])
            # appearance[tuple(ex)] += 1
            
        x1 = subgraph_size*2 - len(list(set(tmp1)))    
        sub_total.append(x1)
        tmp2.extend(tmp1)
        tmp1 = []
        if len(tmp2) == batch_size:
            y1 = batch_size - len(list(set(tmp2)))
            batch_total.append(y1)
            tmp2 = []
    
    for example in total_subgraph:
        appearance[tuple(example)] += 1

    total_subgraph = [{'head_id': head, 'relation': rel, 'tail_id': tail}
                      for head, rel, tail in total_subgraph]
    print(len(total_subgraph))
    return total_subgraph, appearance, sub_total, batch_total
  

import datetime
if __name__ == "__main__":
    import os
    import numpy as np

    train_path = '/home/youminkk/Model_Experiment/2_SubGraph/4_RWR_weighted/data/WN18RR/valid.txt.json'

    
    obj = BFS(train_path)
    batch_size = 1024
    subgraph = 512
    step_size = 169

    sd = time.time()
    bfs_dict = BFS_Dictionary(obj, 30)
    ed = time.time()
    print("Time for Building Path Dictionary: {}".format(datetime.timedelta(seconds = ed - sd)))
    

    pkl_path = '/home/youminkk/Model_Experiment/2_SubGraph/7_BFS/data/WN18RR/valid_bfs.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(bfs_dict, f)
    
    keys = list(bfs_dict.keys())
    x = random.sample(keys, 1)
    x = x[0]
    print(bfs_dict[x][:5]) 

