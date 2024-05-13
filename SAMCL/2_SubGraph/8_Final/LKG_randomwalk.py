from collections import defaultdict, deque
from typing import List, Dict, Tuple
from logger_config import logger
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

def build_appearance(path:str):
    data = json.load(open(path, 'r', encoding='utf-8'))
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
    

def process_triple(example, obj, distribution):
    head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
    appearance = defaultdict(int)
    obj.randomwalk_for_sampling(head_id, relation, tail_id, 30, 10, distribution, appearance)
    return appearance

def process_triple_final(data, example, obj, k_steps, num_iter, distribution, subgraph_size):
    head_id, relation, tail_id = example[0], example[1], example[2]
    subgraph = obj.biased_randomwalk(head_id, relation, tail_id, k_steps, num_iter, distribution, subgraph_size)
    if len(subgraph) < subgraph_size:
        margin = min(subgraph_size, k_steps * num_iter)
        tmp = random.sample(data, margin)
        tmp = [(ex['head_id'], ex['relation'], ex['tail_id']) for ex in tmp]
        subgraph.extend(tmp)
    subgraph[0] = (head_id, relation, tail_id) # first triple in the subgraph list must be a center triple 
    subgraph = subgraph[:subgraph_size]
    assert len(subgraph) == subgraph_size

    return subgraph

def Path_Dictionary_for_LKG1(train_path, obj, distribution, subgraph_size):
    data = json.load(open(train_path, 'r', encoding='utf-8'))

    s = time.time()
    selected_candidates = Selecting_candidates(obj.Graph, data)
    e = time.time()

    num_candidates = len(data) // subgraph_size
    
    logger.info(f"The number of Final Candidates: {num_candidates}")
    logger.info(f"Time for Selecting: {datetime.timedelta(seconds=e-s)}")

    batch_size = 10000
    num_batches = (len(selected_candidates) + batch_size - 1) // batch_size
    logger.info(f"Total Batch in LKG1 Function: {num_batches}")

    pre_appearance = defaultdict(int)
    cnt = 0
    logger.info(f"Step 1. Extracting Valuable Center Triples!!")
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(selected_candidates))
        batch_candidates = selected_candidates[start_index:end_index]
        logger.info(f"Start Biased Randomwalk with Restart for batch {i+1}!!")
        s = time.time()
        for example in batch_candidates:
            batch_appearance = process_triple(example, obj, distribution)
            for key, value in batch_appearance.items():
                pre_appearance[key] += value
            cnt += 1
            if cnt % 5000 == 0:
                logger.info(f"Center Triple Done: {cnt}")
        logger.info(f"Processed {len(batch_candidates)} triples in batch {i+1}")
        
        del batch_candidates
        gc.collect()

        e = time.time()
        logger.info(f"Time for Biased Randomwalk with Restart for batch {i+1}: {datetime.timedelta(seconds=e-s)}")   
    
    return pre_appearance, data


def Path_Dictionary_for_LKG2(data, appearance, obj, k_steps, num_iter, distribution, subgraph_size):
    logger.info(f"Step 2. Subgraph Sampling with Biased Random Walk with Restart!!")
    subgraph_dict = defaultdict(list)
    num_candidates = len(data) // subgraph_size
    s = time.time()
    sorted_triples = sorted(appearance.items(), key=lambda x: x[1], reverse=True)
    sorted_triples = [key_value[0] for key_value in sorted_triples]
    
    final_candidates = sorted_triples[:num_candidates]

    assert len(final_candidates) == num_candidates

    logger.info(f"Center Triples: {len(final_candidates)}")
    
    cnt = 0
    for triple in final_candidates:
        subgraph = process_triple_final(data, triple, obj, k_steps, num_iter, distribution, subgraph_size)
        subgraph_dict[triple].extend(subgraph)
        cnt += 1
        if cnt % 5000 == 0:
            logger.info(f"Done: {cnt}")
    e = time.time()
    logger.info(f"Done building Subgraph Dictionary: {datetime.timedelta(seconds = e - s)}")

    result = []
    result.append(subgraph_dict) # Subgraph Dictionary = result[0]
    result.append(final_candidates) # Center Triples = result[1]

    del subgraph_dict
    del final_candidates

    return result

def Path_Dictionary_for_LKG3(data, appearance, total_appearance, obj, k_steps, num_iter, distribution, subgraph_size, phases):
    logger.info(f"Step 2. Subgraph Sampling with Biased Random Walk with Restart!!")
    subgraph_dict = defaultdict(list)
    num_candidates = len(data) // subgraph_size
    s = time.time()
    sorted_triples = sorted(appearance.items(), key=lambda x: x[1], reverse=True)
    sorted_triples = [key_value[0] for key_value in sorted_triples]
    
    final_candidates = sorted_triples[:num_candidates]
    logger.info(f"Center Triples for 1st Phase: {len(final_candidates)}")
    cnt = 0
    
    center_list = []
    center_list.extend(final_candidates)

    assert len(final_candidates) == num_candidates
    assert len(center_list) % num_candidates == 0

    cnt = 0
    # 1st Phase
    for triple in final_candidates:
        logger.info(f"cnt: {cnt}")
        subgraph = process_triple_final(data, triple, obj, k_steps, num_iter, distribution, subgraph_size)

        for ex in subgraph:
            total_appearance[ex] += 1

        subgraph_dict[triple].extend(subgraph)
        cnt += 1
        if num_candidates < 100:
            if cnt % 10 == 0:
                logger.info(f"Done {cnt} Triples in Phase 1")
        else:
            if cnt % 5000 == 0:
                logger.info(f"Done: {cnt}")
           
    print("Done Phase: 1")

    # 2nd Phase ~
    for phase in range(phases - 1):
        sorted_triples = sorted(total_appearance.items(), key = lambda x: x[1], reverse=True)
        sorted_triples = [key_value[0] for key_value in sorted_triples]

        candidates_phase = sorted_triples[:num_candidates]

        assert len(candidates_phase) == num_candidates

        center_list.extend(candidates_phase)

        for triple in candidates_phase:
            if not triple in subgraph_dict:
                subgraph = process_triple_final(data, triple, obj, k_steps, num_iter, distribution, subgraph_size)
                assert len(subgraph) == subgraph_size
                subgraph_dict[triple].extend(subgraph)
                for ex in subgraph:
                    total_appearance[ex] += 1
            else:
                subgraph = subgraph_dict[triple]
                for ex in subgraph:
                    total_appearance[ex] += 1
        print(f"Done Phase: {phase + 2}")

    e = time.time()
    logger.info(f"Done building Subgraph Dictionary: {datetime.timedelta(seconds = e - s)}")

    result = []
    result.append(subgraph_dict)
    result.append(center_list)

    assert len(center_list) == (num_candidates * phases)
    assert len(list(set(center_list))) == len(list(subgraph_dict.keys()))

    del subgraph_dict
    del center_list

    return result
   

def Making_Subgraph_for_LKG(subgraph_dict, centers):
    total_subgraph = []
    for center in centers:
        total_subgraph.extend(subgraph_dict[center])
        
    total_subgraph = [{'head_id': head, 'relation': relation, 'tail_id': tail} for head, relation, tail in total_subgraph]
    return total_subgraph

import os
import re

def main(base_dir, dataset, k_steps, num_iter, distribution, phase, subgraph_size, mode):
    if phase == 1:
        ## Step 1
        s = time.time()
        data_file = f'{mode}.txt.json'
        data_path = os.path.join(base_dir, dataset, data_file)

        obj = Biased_RandomWalk(data_path, distribution)

        total_appearance, data = Path_Dictionary_for_LKG1(data_path, obj, distribution, subgraph_size)

        pkl_file = f'{mode}_{distribution}_appearance.pkl'
        pkl_path = os.path.join(base_dir, dataset, pkl_file)

        with open(pkl_path, 'wb') as f:
            pickle.dump(total_appearance, f)
        print(f"Counted Appearance Dictionary saved to {pkl_path}")
        e = time.time()
        logger.info("Time for building appearance: {}".format(datetime.timedelta(seconds=e-s)))

        ## Step 2
        Subgraph_List = Path_Dictionary_for_LKG2(data, total_appearance, obj, k_steps, num_iter, distribution, subgraph_size)
        # Subgraph_List[0] = subgraph_dictionary
        # Subgraph_List[1] = center_triple_list

        dict_file = f'{mode}_{distribution}_{k_steps}_{num_iter}.pkl'
        dict_path = os.path.join(base_dir, dataset, dict_file)
        with open(dict_path, 'wb') as f:
            pickle.dump(Subgraph_List, f)
        print(f"BRWR Dictionary saved to {dict_path}")

    else:
        s = time.time()
        data_file = f'{mode}.txt.json'
        data_path = os.path.join(base_dir, dataset, data_file)
        obj = Biased_RandomWalk(data_path, distribution)
    
        data_total = json.load(open(data_path, 'r', encoding='utf-8'))
        total_appearance = defaultdict(int)
        for ex in data_total:
            triple = (ex['head_id'], ex['relation'], ex['tail_id'])
            total_appearance[triple] = 0

        del data_total

        pre_appearance, data = Path_Dictionary_for_LKG1(data_path, obj, distribution, subgraph_size)

        pkl_file = f'{mode}_{distribution}_appearance.pkl'
        pkl_path = os.path.join(base_dir, dataset, pkl_file)

        with open(pkl_path, 'wb') as f:
            pickle.dump(pre_appearance, f)
        print(f"Counted Appearance Dictionary saved to {pkl_path}")
        print(f"This counting is only for extracting initial center triples.")
        e = time.time()
        logger.info("Time for building appearance: {}".format(datetime.timedelta(seconds=e-s)))

        ## Step 2
        Subgraph_List = Path_Dictionary_for_LKG3(data, pre_appearance, total_appearance, obj, k_steps, num_iter, distribution, subgraph_size, phase)
        # Subgraph_List[0] = subgraph_dictionary
        # Subgraph_List[1] = center_triple_list

        dict_file = f'{mode}_{distribution}_{k_steps}_{num_iter}.pkl'
        dict_path = os.path.join(base_dir, dataset, dict_file)
        with open(dict_path, 'wb') as f:
            pickle.dump(Subgraph_List, f)
        print(f"BRWR Dictionary saved to {dict_path}")

        

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
