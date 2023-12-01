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

from config import args


class RandomWalk:
    def __init__(self, train_path: str):
        print('Start to build link graph!!!')
        # id -> {(relation, id)}
        self.graph = defaultdict(set)
        
        # Directed Graph
        self.directed_graph = defaultdict(set)
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        self.head = []
        self.tail = []

        for ex in examples:
            head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
            
            # Add to graph with all details
            if head_id not in self.graph:
                self.graph[head_id] = set()
                self.directed_graph[head_id] = set()
            # Link Graph(Undirected Graph)
            self.graph[head_id].add((head_id, relation, tail_id)) 
            # Extraction Graph(Directed Graph) 
            self.directed_graph[head_id].add((head_id, relation, tail_id))

            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add((tail_id, relation, head_id))           
        
        print('Done building link graph with {} nodes'.format(len(self.graph)))
        print('Done building DIRECTED graph with {} nodes'.format(len(self.directed_graph)))
        

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        # Fetch immediate neighbors for the given tail_id
        neighbors = [item[2] for item in self.graph.get(tail_id, set())]

        # List to collect neighbors found in subsequent hops
        distant_neighbors = []

        # Use recursion to fetch neighbors of neighbors
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))

        # Return unique neighbor IDs by converting to set and then back to list
        return list(set(neighbors + distant_neighbors))
    

    def random_walk(self, head_id, relation, tail_id, k_steps, max_samples, num_iter):
        # Initialize variables and data structures
        head, tail = head_id, tail_id
        graph = self.directed_graph
        all_entities = list(graph.keys())
        current_entity = head_id
        samples = []
        negative_tails = set()
        tail_neighbors = set(self.get_neighbor_ids(tail, 1))
        tail_neighbors.add(tail)
        for _ in range(num_iter):  # Perform random walk 1000 times
            current_entity = head
            for _ in range(k_steps):
                neighbors = self.get_neighbor_ids(current_entity, 1)

                if not neighbors:  # If there are no neighbors, break
                    break
                candidate = random.choice(list(neighbors))
                current_entity = candidate
            # Use the entity reached after 5 steps as a candidate negative tail
            negative_tails.add(candidate)
            negative_tails = negative_tails - tail_neighbors
        
        # Ensure that we have a list of exactly 511 unique negative tails
        while len(negative_tails) < max_samples:
            while True:
                random_entity = random.choice(all_entities)
                if random_entity not in negative_tails:
                    negative_tails.add(random_entity)
                    break

        negative_tails = list(negative_tails)
        negative_tails = random.sample(negative_tails, max_samples)
        
        sampling_count = {}
        # Create samples with the selected negative tails
        for tmp in negative_tails:
            samples.append({'head_id': head, 'relation': relation, 'tail_id': tmp})
            
        # Insert the positive sample at the beginning
        samples.insert(0, {'head_id': head, 'relation': relation, 'tail_id': tail})
        if (head, relation) not in sampling_count:
            sampling_count[(head, relation)] = 1
            sampling_count[(tail, 'inverse {}'.format(relation))] = 1
        else:
            sampling_count[(head, relation)] += 1
            sampling_count[(tail, 'inverse {}'.format(relation))] += 1
            
        return samples, sampling_count


import json
import multiprocessing
import random
import time
import datetime
from collections import defaultdict

def random_walk_for_example(example, G):
    head_id = example['head_id']
    relation = example['relation']
    tail_id = example['tail_id']
    # positive_samples, sampling_count = G.random_walk(head_id, relation, tail_id, 5, 15, 2000)
    positive_samples, sampling_count = G.random_walk(head_id, relation, tail_id, args.randomwalk_step, (args.negative_size -1), args.iteration)
    return positive_samples

def process_data(data, G):
    all_positive_samples = []
    for example in data:
        positive_samples = random_walk_for_example(example, G)
        all_positive_samples.extend(positive_samples)
    return all_positive_samples

def randomwalk_sampling(data, graph, num_process=45):
    pool = multiprocessing.Pool(processes = num_process)
    chunks = [data[i:i + len(data) // num_process] for i in range(0, len(data), len(data) // num_process)]
    results = pool.starmap(process_data, [(chunk, graph) for chunk in chunks])
    all_positive_samples = []
    for result in results:
        all_positive_samples.extend(result)
    pool.close()
    pool.join()
    return all_positive_samples


