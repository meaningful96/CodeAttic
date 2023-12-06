"""
Created on meaningful96

DL Project
"""

from collections import defaultdict, deque
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
from functools import partial
from logger_config import logger

import datetime
import random
import json
import time


def build_graph(train_path: str):
    # id -> {(relation, id)}
    graph = defaultdict(set)

    # Directed Graph
    directed_graph = defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding='utf-8'))
    appearance = {}
    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0
        # Add to graph with all details
        if head_id not in graph:
            graph[head_id] = set()
            directed_graph[head_id] = set()
        # Link Graph(Undirected Graph)
        graph[head_id].add((head_id, relation, tail_id))
        # Extraction Graph(Directed Graph)
        directed_graph[head_id].add((head_id, relation, tail_id))

        if tail_id not in graph:
            graph[tail_id] = set()
        graph[tail_id].add((tail_id, relation, head_id))

    entities = list(graph.keys())
    # print('Done building link graph with {} nodes'.format(len(graph)))
    # print('Done building DIRECTED graph with {} nodes'.format(len(directed_graph)))
    return graph, directed_graph, appearance, entities


class RandomWalk:
    def __init__(self, train_path: str):
        self.Graph, self.diGraph, self.appearance, self.entities = build_graph(train_path)

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        # Fetch immediate neighbors for the given tail_id
        neighbors = [item[2] for item in self.Graph.get(tail_id, set())]

        # List to collect neighbors found in subsequent hops
        distant_neighbors = []

        # Use recursion to fetch neighbors of neighbors
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops - 1))

        # Return unique neighbor IDs by converting to set and then back to list
        return list(set(neighbors + distant_neighbors))

    def bfs(self, start: str) -> Dict[str, int]:
        visited = {}
        queue = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()
            if node not in visited:
                visited[node] = depth
                for neighbor in self.Graph.get(node, []):  # Link graph
                    queue.append((neighbor[2], depth + 1))

        return visited

    def departing(self):
        graph = self.Graph  # Fix: Use self.Graph instead of graph
        all_entities = list(self.Graph.keys())
        entities = list(self.diGraph.keys())
        candidates = list(set(all_entities) - set(entities))
        center_length = len(self.bfs(random.choice(list(self.diGraph.keys()))))  # Fix: Use self.bfs
        fully_disconnected = []
        for entity in candidates:
            bfs = self.bfs(entity)  # Fix: Use self.bfs
            if len(bfs.keys()) != center_length:
                fully_disconnected.append(entity)
        disconnected_triple = []
        for entity in fully_disconnected:
            disconnected_triple.extend(list(self.Graph[entity]))  # Fix: Use self.Graph

        return fully_disconnected, disconnected_triple

    def randomwalk(self, entity, k_steps, num_iter):
        graph = self.Graph
        directed_graph = self.diGraph
        all_path = []
        for _ in range(num_iter):
            current_entity = entity
            path = []
            for _ in range(k_steps):
                neighbors = self.get_neighbor_ids(current_entity, 1)
                candidate = random.choice(list(neighbors))
                if current_entity in directed_graph.keys():
                    for triple in directed_graph[current_entity]:
                        if triple[2] == candidate:
                            path.append({'head_id': triple[0], 'relation': triple[1], 'tail_id': triple[2]})

                if current_entity not in directed_graph.keys():
                    for triple in graph[current_entity]:
                        if triple[2] == candidate:
                            path.append({'head_id': triple[2], 'relation': triple[1], 'tail_id': triple[0]})

                current_entity = candidate

            unique_set = {frozenset(d.items()) for d in path}
            unique_path = [dict(fs) for fs in unique_set]

            # Ordering
            unique_path = [{'head_id': d['head_id'], 'relation': d['relation'], 'tail_id': d['tail_id']} for d in
                           unique_path]
            all_path.append(unique_path)

        return all_path

    def build_subgraph(self, path_list, subgraph_size):
        path_list_shuffled = list(path_list)
        random.shuffle(path_list_shuffled)
        appearance = self.appearance
        subgraph = []
        while len(subgraph) < subgraph_size:
            for path in path_list_shuffled:
                subgraph.extend(path)
                for ex in path:
                    key = (ex['head_id'], ex['relation'], ex['tail_id'])
                    if key in appearance:
                        appearance[key] += 1

        subgraph = subgraph[:subgraph_size]

        return subgraph, appearance


def process_partition(entity_partition, rw, k_steps, num_iter, subgraph_size, appearance, disconnected_triple):
    processed_data_partition = defaultdict(list)
    for entity in entity_partition:
        if entity not in processed_data_partition:
            processed_data_partition[entity] = list()
        all_path = rw.randomwalk(entity, k_steps, num_iter)
        processed_data_partition[entity].extend(all_path)

    tmp = []
    for entity in entity_partition:
        if entity not in processed_data_partition:
            processed_data_partition[entity] = list()
        triples = random.sample(disconnected_triple, k_steps)
        for triple in triples:
            tmp.append({'head_id': triple[2], 'relation': triple[1], 'tail_id': triple[0]})

    total_data_partition = []  # Final Outputs
    ent = random.choice(entity_partition)
    while all(value != 0 for value in appearance.values()):
        path_list = processed_data_partition[ent]
        path_list_shuffled = list(path_list)
        random.shuffle(path_list_shuffled)
        subgraph_ent, appearance_ent = rw.build_subgraph(path_list_shuffled, subgraph_size)
        for key in appearance.keys():
            appearance[key] = appearance[key] + appearance_ent[key]
        keys_greater_than_zero = [key for key, value in appearance.items() if value > 0]
        next_triple = random.choice(keys_greater_than_zero)
        next_candidates = [next_triple[0], next_triple[2]]
        ent = random.choice(next_candidates)
        total_data_partition.extend(subgraph_ent)

    return total_data_partition

def process_parallel(path, k_steps, num_iter, subgraph_size, num_processes):
    Graph, diGraph, appearance, entities = build_graph(path)
    rw = RandomWalk(path)
    logger.info("Building Graph Done!!")
    fully_disconnected, disconnected_triple = rw.departing()

    normal_entities = list(set(entities) - set(fully_disconnected))

    partition_size = len(normal_entities) // num_processes
    entity_partitions = [normal_entities[i:i + partition_size] for i in range(0, len(normal_entities), partition_size)]

    with Manager() as manager:
        shared_appearance = manager.dict(appearance)
        pool = Pool(num_processes)

        logger.info("Start to build paths list!!")  # Additional print statement
        
        start_parallel = time.time()
        results = pool.starmap(
            process_partition,
            [
                (entity_partition, rw, k_steps, num_iter, subgraph_size, shared_appearance, disconnected_triple)
                for entity_partition in entity_partitions
            ]
        )
        pool.close()
        pool.join()
        end_parallel = time.time()
        
    # Combine results from different processes
    total_data = [item for sublist in results for item in sublist]

    print("Time for parallel processing: {}".format(datetime.timedelta(seconds=end_parallel - start_parallel)))

    return total_data



start_total = time.time()
path = '/home/youminkk/Model_Experiment/2_SubGraph/0_RandomWalk/data/WN18RR/train.txt.json'
final_parallel = process_parallel(path, 7, 100, 32, 40)
print(len(final_parallel)//1024 + 1)
end_total = time.time()
print("Total time for the entire process: {}".format(datetime.timedelta(seconds=end_total - start_total)))


