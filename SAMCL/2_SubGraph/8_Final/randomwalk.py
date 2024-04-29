from collections import defaultdict, deque
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
from logger_config import logger

import multiprocessing
import networkx as nx
import numpy as np
import datetime
import random
import pickle
import time
import json
import copy

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

def build_nxGraph(train_data):
    graph = nx.Graph()
    for ex in train_data:
        graph.add_node(ex['head_id'], label=ex['head'])
        graph.add_node(ex['tail_id'], label=ex['tail'])
        graph.add_edge(ex['head_id'], ex['tail_id'], relatio=ex['relation'])
    return graph


def rearrange_list(input_list):
    output_list = [(h, r, t) for (t, r, h) in input_list]
    return output_list

def Shuffle(input_list):
    out_list = []
    for sublist in input_list:
        random.shuffle(sublist)
        out_list.append(sublist)
    return out_list

def weighted_random_selection(prob_distribution):
    # Generating a list of indices based on the probability distribution
    indices = list(range(len(prob_distribution)))
    
    # Performing weighted random selection for a single trial
    selected_index = random.choices(indices, weights=prob_distribution, k=1)[0]

    return selected_index



class Biased_RandomWalk:
    def __init__(self, train_path:str):
        self.Graph, self.Graph_tail, self.diGraph, self.appearance, self.entities = build_graph(train_path)
        logger.info("Done building Link Graph with {} nodes".format(len(self.Graph)))
        logger.info("Done building Directed Graph with {} nodes".format(len(self.diGraph)))

        self.train_dataset = json.load(open(train_path, 'r', encoding='utf-8'))

        # NetworkX Graph
        self.nxGraph = build_nxGraph(self.train_dataset)
        self.degree_prob, self.degree_list = defaultdict(list), defaultdict(list)
        
        self.train_dataset = [] # no need to use

        for entity in self.entities:
            neighbors = self.get_neighbor_ent_ids(entity)
            prob_proportional = list(np.zeros(len(neighbors)))
            degree_ent = list(np.zeros(len(neighbors)))
            for i, neighbor in enumerate(neighbors):
                prob_proportional[i] = self.nxGraph.degree(neighbor)
                degree_ent[i] = neighbor
            prob_antithetical = [1/degree for degree in prob_proportional]
            self.degree_prob[entity].extend([prob_proportional, prob_antithetical])
            self.degree_list[entity].extend(degree_ent)            
                    

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

    def randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, distribution: str) -> List[list]:
        graph = self.Graph
        directed_graph = self.diGraph
        degree_prob = self.degree_prob # [0]: Degree Proportional, [1]: Degree Antithetical
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        subgraph_candidates = []         
        nh, nt = len(self.get_neighbor_ent_ids(head_id)), len(self.get_neighbor_ent_ids(tail_id))
        center_ent = [head_id, tail_id]    
        center_pro, center_ant = [nh, nt], [1/nh, 1/nt]

        # Hard Negative is the close neighbors from the center triple
        # For example, if the triple is placed far away from the center triple,
        # It's not Hard negative. It is called easy negative
        # The Step number is for the boundary and the reset probabaility is the half of the step number 

        prob_reset = k_steps//2 # Mean of the path length 
        r_prob = 1 / prob_reset
        s_prob = 1 - r_prob

        # Step 1. Selecting the Start Point
        cnt =  0
        while cnt <= num_iter:
            
            
            # Uniform Distribution
            if distribution == "uniform":
                current_entity = random.choice([head_id, tail_id])
            
            
            # Degree Proportional
            if distribution == 'proportional':
                current_entity = center_ent[weighted_random_selection(center_pro)]
            
            
            # Degree Antithetical
            if distribution == 'antithetical': 
                current_entity = center_ent[weighted_random_selection(center_ant)]

            # Step 2. Random Walk with Restart(RWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            
            triples_list = []
            triples_list.append(center_triple)
            visited = set()
            visited.add(current_entity)
            candidate = None
            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    subgraph_candidates.extend(triples_list)
                    break
                
                # Walking toward the neighbors
                if standard == 1:

                    neighbors = self.get_neighbor_ent_ids(current_entity)
                                      
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        if distribution == 'uniform':
                            candidate = random.choice(list(neighbors))
                        
                        # Degree Proportional
                        if distribution == 'proportional':
                            candidate_prob = degree_prob[current_entity][0] # Probability
                            candidate_list = self.get_neighbor_ent_ids(current_entity)
                            selected_index = weighted_random_selection(candidate_prob)
                            candidate = candidate_list[selected_index]
                       
                        # Degree Antithetical
                        if distribution == 'antithetical':
                            candidate_prob = degree_prob[current_entity][1] # Probability
                            candidate_list = self.get_neighbor_ent_ids(current_entity)
                            selected_index = weighted_random_selection(candidate_prob)
                            candidate = candidate_list[selected_index]

                    visited.add(candidate)
                    if current_entity in directed_graph.keys():
                        for triple in directed_graph[current_entity]:
                            if triple[2] == candidate:
                                triples_list.append(triple)
                                append = True
                                
                        if append == False:
                            for triple in directed_graph[candidate]:
                                if triple[2] == current_entity:
                                    triples_list.append(triple)   
                                    append = True
                
                    if current_entity not in directed_graph.keys():
                        for triple in directed_graph[candidate]:
                            if triple[2] == current_entity:
                                triples_list.append(triple)
                                append = True
                    
                    assert append == True, "No triple is appended!!"
                    current_entity = candidate                                
                
                standard = weighted_random_selection(prob_list)
            subgraph_candidates.extend(triples_list)
            subgraph_candidates = list(set(subgraph_candidates))
            cnt += len(subgraph_candidates)
        
        subgraph_candidates = subgraph_candidates[:num_iter]
        return subgraph_candidates


def Path_Dictionary(train_path, k_steps, num_iter, obj, num_process, distribution):
    train_data = json.load(open(train_path, 'r', encoding='utf-8'))
    triple_dict = defaultdict(list)
    
    data = train_data
    chunk_size = len(data) // num_process
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    logger.info("Start Biased Randomwalk with Restart!!")
    with multiprocessing.Pool(num_process) as pool:
        results = pool.starmap(process_data_chunk, [(chunk, data, obj, k_steps, num_iter, train_data, distribution) for chunk in chunks])

    for chunk_result in results:
        for key, value in chunk_result.items():
            triple_dict[key].extend(value)
    return triple_dict

def random_sample(train_data):
    sample = random.sample(train_data, 1024)
    
    all_path = []
    for ex in sample:
        all_path.append((ex['head_id'], ex['relation'], ex['tail_id']))
    all_path = list(set(all_path))
    
    return all_path

def process_data_chunk(chunk, data, obj, k_steps, num_iter, train_data, distribution):
    chunk_triple_dict = defaultdict(list)

    for example in chunk:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        center_triple = (head_id, relation, tail_id)
        all_path = obj.randomwalk(head_id, relation, tail_id, k_steps, num_iter, distribution)
        if len(all_path) < 1024:
            tmp = random_sample(train_data)
            all_path.extend(tmp)
        chunk_triple_dict[center_triple].extend(all_path)

    return chunk_triple_dict

def Making_Subgraph(path_dict, candidates, subgraph_size, appearance, batch_size):
    total_subgraph = []
    batch_total, sub_total = [], []
    tmp1, tmp2 = [], []
    p = len(candidates)
    # ent_dict, rel_dict = idx2ent(dict1), idx2ent(dict2)
    for candidate in candidates:
        subgraph = []

        # Extract the all subgraph candidates
        path_list = path_dict[candidate]    
        
        # Shuffle the candidates list
        random.shuffle(path_list)   

        # Using Set()
        hr_set = set()

        # add center triple
        subgraph.append(candidate)
        hr_set.add(candidate[0])
        hr_set.add(candidate[2])
        
        for triple in path_list:
            h, t = triple[0], triple[2]
            if h not in hr_set and t not in hr_set:
                subgraph.append(triple)
                hr_set.add(h)

                hr_set.add(t)

            if len(subgraph) >= subgraph_size:
                break

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
            appearance[tuple(ex)] += 1
            
        x1 = subgraph_size*2 - len(list(set(tmp1)))    
        sub_total.append(x1)
        tmp2.extend(tmp1)
        tmp1 = []
        if len(tmp2) == batch_size:
            y1 = batch_size - len(list(set(tmp2)))
            batch_total.append(y1)
            tmp2 = []


    total_subgraph = [{'head_id': head, 'relation': rel, 'tail_id': tail}
                      for head, rel, tail in total_subgraph]

    return total_subgraph, appearance, sub_total, batch_total
   

import os
import pickle
import time
import datetime
import argparse

def main(base_dir, dataset, k_step, n_iter, num_cpu, distribution, mode):
    train_file = f'{mode}.txt.json'
    train_path = os.path.join(base_dir, dataset, train_file)

    obj = Biased_RandomWalk(train_path)

    sd = time.time()
    # Assuming `subgraph` is defined or imported elsewhere
    path_dict = Path_Dictionary(train_path, k_step, n_iter, obj, num_cpu, distribution)
    ed = time.time()
    logger.info("Time for Building Path Dictionary: {}".format(datetime.timedelta(seconds=ed - sd)))

    pkl_file = f'{mode}_{distribution}_{k_step}_{n_iter}.pkl'
    pkl_path = os.path.join(base_dir, dataset, pkl_file)

    with open(pkl_path, 'wb') as f:
        pickle.dump(path_dict, f)
    print(f"Path Dictionary saved to {pkl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save path dictionary.")
    parser.add_argument("--base-dir", type=str, required=True, help="base path")
    parser.add_argument("--dataset", type=str, choices=['WN18RR', 'FB15k237', 'wiki5m_ind', 'wiki5m_trans'], required=True, help="Dataset name")
    parser.add_argument("--k-step", type=int, required=True, help="Number of steps for the random walk")
    parser.add_argument("--n-iter", type=int, required=True, help="Number of iterations for the random walk")
    parser.add_argument("--num-cpu", type=int, required=True, help="Number of CPUs for parallel processing")
    parser.add_argument("--distribution", type=str, choices=['uniform', 'proportional', 'antithetical'], required=True, help="Dataset name")
    parser.add_argument("--mode", type=str, choices=['train', 'valid'], required=True, help="mode")
    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.k_step, args.n_iter, args.num_cpu, args.distribution, args.mode)
