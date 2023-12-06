from collections import defaultdict, deque
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
from logger_config import logger

import datetime
import random
import time
import json


def build_graph(train_path:str):
    Graph, diGraph = defaultdict(set), defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding = 'utf-8'))
    appearance = {}

    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0

        if head_id not in Graph:
            Graph[head_id] = set()
            diGraph[head_id] = set()
        Graph[head_id].add((head_id, relation, tail_id))
        diGraph[head_id].add((head_id, relation, tail_id))
        
        if tail_id not in Graph:
            Graph[tail_id] = set()
        Graph[tail_id].add((tail_id, relation, head_id))

    entities = list(Graph.keys())
    # logger.info("Done building Link Graph with {} nodes".format(len(Graph)))
    # logger.info("Done building Directed Graph with {} nodes".format(len(diGraph)))

    return Graph, diGraph, appearance, entities


class RandomWalk:
    def __init__(self, train_path:str):
        self.Graph, self.diGraph, self.appearance, self.entities = build_graph(train_path)
        logger.info("Done building Link Graph with {} nodes".format(len(self.Graph)))
        logger.info("Done building Directed Graph with {} nodes".format(len(self.diGraph)))

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        # Fetch immediate neighbors for the given tail_id
        neighbors = [item[2] for item in self.Graph.get(tail_id, set())]

        # List to collect neighbors found in subsequent hops
        distant_neighbors = []

        # Use recursion to fetch neighbors of neighbors
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))

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

    def Departing(self):
        graph = self.Graph  
        all_entities = list(self.Graph.keys())
        entities = list(self.diGraph.keys())
        candidates = list(set(all_entities) - set(entities))
        center_length = len(self.bfs(random.choice(list(self.diGraph.keys()))))  
        fully_disconnected = []
        for entity in candidates:
            bfs = self.bfs(entity)  
            if len(bfs.keys()) != center_length:
                fully_disconnected.append(entity)
        disconnected_triple = []
        for entity in fully_disconnected:
            disconnected_triple.extend(list(self.Graph[entity]))  

        return fully_disconnected, disconnected_triple

    def randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int) -> List[list]:
        graph = self.Graph
        directed_graph = self.diGraph
        all_path = []
        center_triple = {'head_id': head_id, 'relation': relation, 'tail_id': tail_id}
        
        ## for 'head' with center triple
        for _ in range(num_iter // 2):
            path = []
            current_entity = head_id
            path.append(center_triple)
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
                
        ## for 'tail' with center triple
        for _ in range(num_iter // 2):
            path = []
            current_entity = tail_id
            path.append(center_triple)
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

    def Combinating_Subgraph(self, path_list, subgraph_size):
        path_list = path_list
        random.shuffle(path_list)
        appearance = self.appearance
        subgraph = []
        while len(subgraph) < subgraph_size:
            for path in path_list:
               subgraph.extend(path)
               for ex in path:
                   key = (ex['head_id'], ex['relation'], ex['tail_id'])
                   if key in appearance:
                       appearance[key] += 1
        subgraph = subgraph[:subgraph_size]
        
        return subgraph, appearance


def Path_Dictionary(train_path, k_steps, num_iter, obj):
    data = json.load(open(train_path, 'r', encoding = 'utf-8'))
    triple_dict = defaultdict(list)
    fully_disconnected, disconnected_triple = obj.Departing()

    for example in data:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        
        # Skip disconnected triples
        if (head_id, relation, tail_id) in disconnected_triple:
            continue

        all_path = obj.randomwalk(head_id, relation, tail_id, k_steps, num_iter)
        if ((head_id, relation, tail_id)) not in triple_dict:
            triple_dict[(head_id, relation, tail_id)] = list()
        triple_dict[(head_id, relation, tail_id)].extend(all_path)
    
    # Add disconnecte_triple path in triple_dict
    for example in disconnected_triple:
        head_id, relation, tail_id = example[0], example[1], example[2]
        
        if ((head_id, relation, tail_id)) not in triple_dict:
            triple_dict[(head_id, relation, tail_id)] = list()
        all_path = []
        for i in range(num_iter):
            path = []
            path.append({'head_id': head_id, 'relation': relation, 'tail_id': tail_id})
            tmp = list(random.sample(disconnected_triple, k_steps-1))
            for ex in tmp:
                path.append({'head_id': ex[0], 'relation':ex[1], 'tail_id': ex[2]})
            all_path.append(path)
        
        triple_dict[(head_id, relation, tail_id)].extend(all_path)



if __name__ == "__main__":
    train_path = "/home/youminkk/Model_Experiment/2_SubGraph/1_RandomWalk_Dynamic/data/WN18RR/train.txt.json"
    count = 0
    triple_dict = defaultdict(list)
    data = json.load(open(train_path, 'r', encoding='utf-8'))
    obj = RandomWalk(train_path)
    fully_disconnected, disconnected_triple = obj.Departing()
    print(disconnected_triple)
    for example in data:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        all_path = obj.randomwalk(head_id, relation, tail_id, 5, 2)
        if ((head_id, relation, tail_id)) not in triple_dict:
            triple_dict[(head_id, relation, tail_id)] = list()
        triple_dict[(head_id, relation, tail_id)].extend(all_path)
        count += 1
        if count == 3:
            break
    print(triple_dict)
    for example in disconnected_triple:
        count += 1
        head_id, relation, tail_id = example[0], example[1], example[2]
        
        if ((head_id, relation, tail_id)) not in triple_dict:
            triple_dict[(head_id, relation, tail_id)] = list()
        all_path = []
        for i in range(2):
            path = []
            tmp = list(random.sample(disconnected_triple, 5))
            for ex in tmp:
                path.append({'head_id': ex[0], 'relation':ex[1], 'tail_id': ex[2]})
            all_path.append(path)
        
        triple_dict[(head_id, relation, tail_id)].extend(all_path)
        if count == 6:
            break
    print("@"*200)
    print(triple_dict)   

