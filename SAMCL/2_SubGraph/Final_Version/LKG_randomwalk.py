from collections import defaultdict, deque
from typing import List, Dict, Tuple
from logger_config import logger
from multiprocessing import Pool

import multiprocessing
import networkx as nx
import numpy as np
import datetime
import argparse
import random
import pickle
import time
import json
import gc
import os
import re

def build_nxGraph(path:str):
    s = time.time()
    Graph = nx.Graph()
    diGraph = nx.DiGraph()
    data = json.load(open(path, 'r', encoding='utf-8'))
    for ex in data:
        h,r,t = ex['head_id'], ex['relation'], ex['tail_id']
        Graph.add_node(h)
        Graph.add_node(t)
        Graph.add_edge(h,t, relation=r)

        diGraph.add_node(h)
        diGraph.add_node(t)
        diGraph.add_edge(h, t, relation=r)

    del data
    # gc.collect()

    entities = list(Graph.nodes())
    e = time.time() 

    logger.info("Done building NetworkX Graph: {}".format(datetime.timedelta(seconds = e - s)))
    return Graph, diGraph, entities

def build_appearance(data):
    appearance = defaultdict(int)
    for ex in data:
        h,r,t = ex['head_id'], ex['relation'], ex['tail_id'] 
        triple = (h,r,t)
        appearance[triple] = 0

    del data

    return appearance

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
    def __init__(self, train_path:str, distribution:str):
        s = time.time()

        self.Graph, self.diGraph, self.entities = build_nxGraph(train_path)

        if distribution == 'proportional':
            self.degree_prob = defaultdict(list)        
            for entity in self.entities:
                neighbors = list(self.Graph.neighbors(entity))
                prob_proportional = list(np.zeros(len(neighbors)))
                for i, neighbor in enumerate(neighbors):
                    d = self.Graph.degree(neighbor)
                    prob_proportional[i] = d
                    self.degree_prob[entity].append(prob_proportional)

        if distribution == 'antithetical':
            self.degree_prob = defaultdict(list)        
            for entity in self.entities:
                neighbors = list(self.Graph.neighbors(entity))
                prob_antithetical = list(np.zeros(len(neighbors)))
                for i, neighbor in enumerate(neighbors):
                    d = self.Graph.degree(neighbor)
                    prob_antithetical[i] = 1/d
                    self.degree_prob[entity].append(prob_antithetical)

        del self.entities

        gc.collect()
        self.cnt = 0
        e = time.time()
        logger.info("Done Class Initilaization: {}".format(datetime.timedelta(seconds=e-s)))
        
    def find_triple(self, diGraph, e1, e2):
        if diGraph.has_edge(e1, e2):
            relation = diGraph.edges[e1, e2]['relation']
            triple = (e1, relation, e2)
        elif diGraph.has_edge(e2, e1):
            relation = diGraph.edges[e2, e1]['relation']
            triple = (e2, relation, e1)
        return triple

    def randomwalk_for_sampling(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, distribution: str, appearance: dict):
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        nh, nt = len(list(self.Graph.neighbors(head_id))), len(list(self.Graph.neighbors(tail_id)))
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
        for _ in range(num_iter):            
            # Uniform Distribution
            if distribution == "uniform":
                current_entity = random.choice(center_ent)
            # Degree Proportional
            if distribution == 'proportional':
                current_entity = center_ent[weighted_random_selection(center_pro)]            
            # Degree Antithetical
            if distribution == 'antithetical': 
                current_entity = center_ent[weighted_random_selection(center_ant)]


            # Step 2. Random Walk with Restart(RWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            cnt_center = 0
            if cnt_center == 0:
                if center_triple not in appearance:
                    appearance[center_triple] = 1
            cnt_center += 1
            
            margin = 100
            visited = set()
            visited_triple = set()
            visited.add(current_entity)
            candidate = None
            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    break
                # Walking toward the neighbors
                if standard == 1:
                    
                    d_current = self.Graph.degree(current_entity)
                    neighbors = list(self.Graph.neighbors(current_entity))
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        if distribution == 'uniform':
                            candidate = random.choice(neighbors)
                        
                        # Degree Proportional
                        if distribution == 'proportional':
                            if d_current <= margin:
                                candidate_prob = self.degree_prob[current_entity][0]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                            elif d_current > margin:
                                candidate = random.sample(neighbors, 1)[0]

                        # Degree Antithetical
                        if distribution == 'antithetical':
                            if d_current <= margin:
                                candidate_prob = self.degree_prob[current_entity][0]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                            elif d_current > margin:
                                candidate = random.sample(neighbors, 1)[0]

                                
                    visited.add(candidate)
                    triple = self.find_triple(self.diGraph, current_entity, candidate)
                    if triple not in visited_triple:
                        visited_triple.add(triple)
                        appearance[triple] = appearance.get(triple, 0) + 1
                    current_entity = candidate                                                
                standard = weighted_random_selection(prob_list)

    def biased_randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, distribution: str, subgraph_size:int):
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        nh, nt = len(list(self.Graph.neighbors(head_id))), len(list(self.Graph.neighbors(tail_id)))
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
        subgraph = []
        cnt = 0
        zero = 0
        
        subgraph.append(center_triple)
        
        while len(subgraph) < subgraph_size and cnt < num_iter:          
            # Uniform Distribution
            if distribution == "uniform":
                current_entity = random.choice(center_ent)
            # Degree Proportional
            if distribution == 'proportional':
                current_entity = center_ent[weighted_random_selection(center_pro)]            
            # Degree Antithetical
            if distribution == 'antithetical': 
                current_entity = center_ent[weighted_random_selection(center_ant)]

            # Step 2. Random Walk with Restart(RWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            
            margin = 100
            visited = set()       
            candidate = None
            
            hr_set = set()
            hr_set.add(head_id)
            hr_set.add(tail_id)
            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    break
                # Walking toward the neighbors
                if standard == 1:
                    
                    d_current = self.Graph.degree(current_entity)
                    neighbors = list(self.Graph.neighbors(current_entity))
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        if distribution == 'uniform':
                            candidate = random.choice(neighbors)
                        
                        # Degree Proportional
                        if distribution == 'proportional':
                            if d_current <= margin:
                                candidate_prob = self.degree_prob[current_entity][0]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                            elif d_current > margin:
                                candidate = random.sample(neighbors, 1)[0]

                        # Degree Antithetical
                        if distribution == 'antithetical':
                            if d_current <= margin:
                                candidate_prob = self.degree_prob[current_entity][0]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                            elif d_current > margin:
                                candidate = random.sample(neighbors, 1)[0]
                                
                    visited.add(candidate)
                    triple = self.find_triple(self.diGraph, current_entity, candidate)
                    if triple[0] not in hr_set and triple[2] not in hr_set:
                        subgraph.append(triple)
                        hr_set.add(triple[0])
                        hr_set.add(triple[2])
                    current_entity = candidate                                                
                standard = weighted_random_selection(prob_list)
            cnt += 1
        return subgraph
      
    
def Selecting_candidates(nxG, data, select_prob=0.25):
    degrees = [nxG.degree(node) for node in nxG.nodes()]
    average_degree = sum(degrees) // len(degrees)
    del degrees

    degree = defaultdict(int)
    for ex in data:
        s = nxG.degree(ex['head_id']) + nxG.degree(ex['tail_id'])
        degree[s] += 1

    defaultdict_sorted = dict(sorted(degree.items()))
    keys = list(defaultdict_sorted.keys())
    values = list(defaultdict_sorted.values())
    threshold = sum(values) // len(values)

    cnt = k = 0
    for i, value in enumerate(values):
        if value <= threshold:
            cnt += 1
            if cnt == 1:
                k = i
        else:
            cnt = 0
        if cnt == 5:
            break

    num = int(sum(values[:k]) * select_prob)
    threshold_degree = keys[k]

    del degree, keys, values

    num_cand = []
    for ex in data:
        s = nxG.degree(ex['head_id']) + nxG.degree(ex['tail_id'])
        if average_degree <= s <= threshold_degree:
            num_cand.append(ex)

    candidates = random.sample(num_cand, min(num, len(num_cand)))
    del num_cand
    random.shuffle(candidates)
    logger.info("Done building selected candidates!!")

    return [{'head_id': ex['head_id'], 'relation': ex['relation'], 'tail_id': ex['tail_id']} for ex in candidates]
    
def process_triple(data, example, obj, k_steps, num_iter, distribution, subgraph_size):
    head_id, relation, tail_id = example[0], example[1], example[2]
    subgraph = obj.biased_randomwalk(head_id, relation, tail_id, k_steps, num_iter, distribution, subgraph_size)
    if len(subgraph) < subgraph_size:
        margin = min(subgraph_size*2, k_steps * num_iter)
        tmp = random.sample(data, margin)
        tmp = [(ex['head_id'], ex['relation'], ex['tail_id']) for ex in tmp]
        subgraph.extend(tmp)
    subgraph[0] = example # first triple in the subgraph list must be a center triple 
    subgraph = subgraph[:subgraph_size]
    assert len(subgraph) == subgraph_size
    assert subgraph[0] == example

    return subgraph

def counting(appearance, subgraph):
    for triple in subgraph:
        if not triple in appearance:
            appearance[triple] = 0
        appearance[triple] += 1
    return appearance

def process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, appearance, subgraph_dict):
    with Pool(processes=4) as pool:
        results = pool.starmap(process_triple, [(data, center, obj, k_steps, num_iter, distribution, subgraph_size) for center in centers])
    for subgraph in results:
        assert len(subgraph) == subgraph_size
        center = subgraph[0]
        appearance = counting(appearance, subgraph)
        subgraph_dict[center].append(subgraph)
    del results
    return appearance, subgraph_dict

def Path_Dictionary_for_LKG(data, appearance, obj, k_steps, num_iter, distribution, subgraph_size, phase):
    logger.info(f"Building Subgraph Dictionary with BRWR!")
    s = time.time()
    subgraph_dict = defaultdict(list)
    total_candidates = len(data) // subgraph_size * phase
    phase_candidates = total_candidates // phase
    logger.info("Total Phase: {}".format(phase))
    logger.info("Total number of center tripels: {}".format(total_candidates))
    logger.info("Center triples per phase: {}".format(phase_candidates))
    cnt = 0
    if phase <= 2:
        num_candidates = phase_candidates // 4

        centers = random.sample(list(appearance.keys()), num_candidates)
        appearance, subgraph_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, appearance, subgraph_dict)
        cnt += len(centers)
        logger.info(f"Done: {cnt}")
        for _ in range(4*phase - 1):
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1], reverse=True)
            centers = [key_value[0] for key_value in sorted_triples[:num_candidates]]
            appearance, subgraph_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, appearance, subgraph_dict)
            cnt += len(centers)
            logger.info(f"Done: {cnt}")
        return appearance, subgraph_dict
    
    elif phase > 2:
        num_candidates = phase_candidates
        centers = random.sample(list(appearance.keys()), num_candidates)
        appearance, subgraph_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, appearance, subgraph_dict)
        cnt += len(centers)
        logger.info(f"Done: {cnt}")
        for _ in range(phase - 1):
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1], reverse=True)
            centers = [key_value[0] for key_value in sorted_triples[:num_candidates]]
            appearance, subgraph_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, appearance, subgraph_dict)
            cnt += len(centers)
            logger.info(f"Done: {cnt}")
        return appearance, subgraph_dict

def Making_Subgraph_for_LKG(subgraph_dict, centers):
    total_subgraph = []
    for center in centers:
        total_subgraph.extend(subgraph_dict[center])
        
    total_subgraph = [{'head_id': head, 'relation': relation, 'tail_id': tail} for head, relation, tail in total_subgraph]
    return total_subgraph

def Making_Subgraph_for_LKG(subgraph_dict, centers):
    total_subgraph = []
    for center in centers:
        total_subgraph.extend(subgraph_dict[center])
        
    total_subgraph = [{'head_id': head, 'relation': relation, 'tail_id': tail} for head, relation, tail in total_subgraph]
    return total_subgraph

import os
import re


def get_degree_dict(subgraph_dict, nxGraph, subgraph_size):
    logger.info("Get DW Dictionary!!")
    s = time.time()
    keys = list(subgraph_dict.keys())
    degree_dict = defaultdict(list)
    for key in keys:
        subgraph_list = subgraph_dict[key]
        for subgraph in subgraph_list:
            dh_list = []
            dt_list = []
            for triple in subgraph:
                dh = nxGraph.degree(triple[0])
                dt = nxGraph.degree(triple[2])
                dh_list.extend([dh, dt])
                dt_list.extend([dt, dh])
            assert len(dh_list) == subgraph_size * 2
            assert len(dt_list) == subgraph_size * 2
            degree_dict[key].append([dh_list, dt_list])
    e = time.time()
    logger.info(f"Time for building DW Dictionary: {datetime.timedelta(seconds=e-s)}")
    return degree_dict

def get_shortest_distance(nxGraph, center, subgraph_list, subgraph_size):
    shortest_lists = []
    for subgraph in subgraph_list:
        sub_list = list(np.zeros(subgraph_size*2))
        assert len(sub_list) == subgraph_size*2
        assert len(subgraph)*2 == len(sub_list)

        head = center[0]
        for i, triple in enumerate(subgraph):
            tail = triple[2]
            try:
                st = nx.shortest_path_length(nxGraph, source=head, target=tail)
                if st == 0:
                    st = 1
            except nx.NetworkXNoPath:
                st = 999
            except nx.NodeNotFound:
                st = 999
            sub_list[2*i] = 1/st
            sub_list[2*i+1] = 1/st
        shortest_list.append(sub_list)
    return shortest_lists, center

def get_spw_dict(subgraph_dict, nxGraph, subgraph_size):
    logger.info("Get SPW Dictionary!!")
    s = time.time()
    centers = list(subgraph_dict.keys())
    total_sw = defaultdict(list)

    with Pool(processes= 8) as pool:
        results = pool.starmap(get_shortest_distance, [(nxGraph, center, subgraph_dict[center], subgraph_size) for center in centers])
    for shortest_lists, center in results:
        total_sw[center] = sub_lists
    e = time.time()

    del results
    del centers

    logger.info(f"Time for building SPW Dictionary: {datetime.timedelta(seconds=e-s)}")
    return total_sw

def main(base_dir, dataset, k_steps, num_iter, distribution, phase, subgraph_size, mode):
    ## Step 1. Define the all paths
    if mode =='valid':
        num_k_steps = k_steps // 2
        num_iteration = num_iter // 2
    elif mode == 'train':
        num_k_steps = k_steps
        num_iteration = num_iter

    s = time.time()
    data_file = f'{mode}.txt.json'
    data_path = os.path.join(base_dir, dataset, data_file) 
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    logger.info("Data Loading Done!!")

    subgraph_out = os.path.join(base_dir, dataset, f"{mode}_{distribution}_{k_steps}_{num_iter}.pkl")
    appear_out = os.path.join(base_dir, dataset, f"{mode}_appearance.pkl")
    degree_out = os.path.join(base_dir, dataset, f"Degree_{mode}.pkl")
    shortest_out = os.path.join(base_dir, dataset, "ShortestPath_{mode}.pkl")

    ## Step 2. BRWR for extracting subgraphs !!
    appearance = build_appearance(data) 
    obj = Biased_RandomWalk(data_path, distribution)
    total_appearance, subgraph_dict = Path_Dictionary_for_LKG(data, appearance, obj, num_k_steps, num_iteration, distribution, subgraph_size, phase)
    """
    subgraph_dict
    {'center1': [[subgraph1], [subgraph2], [subgraph3], ...]
     'center2': [[subgraph1], [subgraph2]]
     'center3': [[subgraph1]]}
     
     subgraph = [(h1,r1,t1), (h2,r2,t2), ...]
    
    """
    with open(subgraph_out, 'wb') as f:
        pickle.dump(subgraph_dict, f)

    with open(appear_out, 'wb') as f:
        pickle.dump(total_appearance, f)
    del appearance
    del total_appearance

    nxGraph = nx.Graph()
    for ex in data:
        nxGraph.add_node(ex['head_id'])
        nxGraph.add_node(ex['tail_id'])
        nxGraph.add_edge(ex['head_id'], ex['tail_id'], relation = ex['relation'])
    del data

    ## Building Degree_weights_dictionary
    degree_dict = get_degree_dict(subgraph_dict, nxGraph, subgraph_size)
    with open(degree_out, 'wb') as f:
        pickle.dump(degree_dict, f)
    del degree_dict

    if mode == 'train':
        spw_dict = get_spw_dict(subgraph_dict, nxGraph, subgraph_size)
        with open(shortest_out, 'wb') as f:
            pickle.dump(shortest_out, f)
        logger.info("Done building SPW Dictionary!!")
    logger.info("Done building BRWR Subgraph Dictionary!!")
    logger.info("Done building DW Dictioanry!!")
    e = time.time()
    logger.info(f"Time: {datetime.timedelta(seconds = e - s)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save path dictionary.")
    parser.add_argument("--base-dir", type=str, required=True, help="Your base directory")    
    parser.add_argument("--k-steps", type=int, default=50, required=True, help="Maximum Path Length")
    parser.add_argument("--n-iter", type=int, default=20, required=True, help="Number of Iteration")
    parser.add_argument("--dataset", type=str, choices=['WN18RR', 'FB15k237', 'wiki5m_ind', 'wiki5m_trans', 'YAGO3-10'], required=True, help="Dataset name")
    parser.add_argument("--distribution", type=str, choices=['uniform', 'proportional', 'antithetical'], required=True, help="Distribution type")
    parser.add_argument("--phase", type=int, required=True, help="Training Phase")
    parser.add_argument("--subgraph-size", type=int, default=512, required=True, help="Subgraph Size")
    parser.add_argument("--mode", type=str, choices=['train', 'valid'], required=True, help="Mode")
    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.k_steps, args.n_iter, args.distribution, args.phase, args.subgraph_size, args.mode)
