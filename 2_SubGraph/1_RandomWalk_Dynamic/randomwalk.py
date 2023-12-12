from collections import defaultdict, deque
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
from logger_config import logger
#from config import args

import multiprocessing
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

def rearrange_list(input_list):
    output_list = [(h, r, t) for (t, r, h) in input_list]
    return output_list

def Shuffle(input_list):
    out_list = []
    for sublist in input_list:
        random.shuffle(sublist)
        out_list.append(sublist)
    return out_list


class RandomWalk:
    def __init__(self, train_path:str):
        self.Graph, self.diGraph, self.appearance, self.entities = build_graph(train_path)
        logger.info("Done building Link Graph with {} nodes".format(len(self.Graph)))
        logger.info("Done building Directed Graph with {} nodes".format(len(self.diGraph)))

        self.train_dataset = json.load(open(train_path, 'r', encoding='utf-8'))

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

    def randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, subgraph_size) -> List[list]:
        graph = self.Graph
        directed_graph = self.diGraph
        center_triple = (head_id, relation, tail_id)
        total_path = []    

        all_path_head = []
        ## for 'head' with center triple
        for _ in range(num_iter // 2):
            path = []
            current_entity = head_id
            path.append(center_triple)
           
            for _ in range(k_steps):
                append = False
                neighbors = self.get_neighbor_ids(current_entity, 1)
                # neighbors = [n for n in neighbors if n != tail_id]
                candidate = random.choice(list(neighbors))
                
                if current_entity in directed_graph.keys():
                    for triple in directed_graph[current_entity]:
                        if triple[2] == candidate:
                            path.append(triple)
                            append = True
                    if append == False:
                        for triple in directed_graph[candidate]:
                            if triple[2] == current_entity:
                                path.append(triple)   
                                append = True
                
                if current_entity not in directed_graph.keys():
                    for triple in directed_graph[candidate]:
                        if triple[2] == current_entity:
                            path.append(triple)
                            append = True
                assert append == True, "No triple is appended!!"
                current_entity = candidate

            x = random.random()
            all_path_head.extend(path)         
        all_path_head = list(set(all_path_head))

        if len(all_path_head) < (subgraph_size // 2):
            random_head = random.sample(self.train_dataset, (subgraph_size//2 - len(all_path_head)))
            random_head = [(ex['head_id'], ex['relation'], ex['tail_id']) for ex in random_head]
            all_path_head.extend(random_head)
        total_path.append(all_path_head)

        all_path_tail = []
        ## for 'tail' with center triple
        for _ in range(num_iter // 2):
            path = []
            current_entity = tail_id
            path.append(center_triple)
           
            for _ in range(k_steps):
                append = False
                neighbors = self.get_neighbor_ids(current_entity, 1)
                # neighbors = [n for n in neighbors if n != head_id]
                candidate = random.choice(list(neighbors))
                
                if current_entity in directed_graph.keys():
                    for triple in directed_graph[current_entity]:
                        if triple[2] == candidate:
                            path.append(triple)
                            append = True
                    if append == False:
                        for triple in directed_graph[candidate]:
                            if triple[2] == current_entity:
                                path.append(triple)   
                                append = True
                
                if current_entity not in directed_graph.keys():
                    for triple in directed_graph[candidate]:
                        if triple[2] == current_entity:
                            path.append(triple)
                            append = True
                assert append == True, "No triple is appended!!"
                current_entity = candidate

            x = random.random()
            all_path_tail.extend(path)
         
        all_path_tail = list(set(all_path_tail))
        if len(all_path_tail) < (subgraph_size // 2):
            random_tail = random.sample(self.train_dataset, (subgraph_size//2 - len(all_path_tail)))
            random_tail = [(ex['head_id'], ex['relation'], ex['tail_id']) for ex in random_tail]
            all_path_tail.extend(random_tail)
   
        total_path.append(all_path_tail)

        return total_path
#"""    
def Making_Subgraph(path_dict, candidates, subgraph_size, appearance):
    total_subgraph = []
    for triple in candidates:
        path_list = path_dict[triple]
        shuffled_paths = Shuffle(path_list)
        subgraph = []
        # print(len(path_list[0]), len(path_list[1]))
        head_subgraph = random.sample(path_list[0], subgraph_size//2)
        tail_subgraph = random.sample(path_list[1], subgraph_size//2)
        
        subgraph.extend(head_subgraph)
        subgraph.extend(tail_subgraph)
        assert subgraph_size == (len(head_subgraph) + len(tail_subgraph))
        total_subgraph.extend(subgraph)
    
    for example in total_subgraph:
        appearance[example] += 1

    total_subgraph = [{'head_id': head, 'relation': rel, 'tail_id': tail}
                      for head, rel, tail in total_subgraph]
    return total_subgraph, appearance
    

#"""

def path_for_disconnected(data, triple, k_steps, num_iter):
    all_path = []

    ## for head
    path_head = []
    for i in range(num_iter//2):
        path = []
        sample = random.sample(data, k_steps)
        for ex in data:
            path.append((ex['head_id'], ex['relation'], ex['tail_id']))
        path.append(triple)
        path_head.extend(path)
    all_path_head = list(set(path_head))

    ## for tail
    path_tail = []
    for i in range(num_iter//2):
        path = []
        sample = random.sample(data, k_steps)
        for ex in data:
            path.append((ex['head_id'], ex['relation'], ex['tail_id']))
        path.append(triple)
        path_tail.extend(path)
    all_path_tail = list(set(path_tail))    

    all_path.append(all_path_head)
    all_path.append(all_path_tail)

    return all_path

"""
# Single CPU

def Path_Dictionary(train_path, k_steps, num_iter, obj):
    data = json.load(open(train_path, 'r', encoding='utf-8'))
    triple_dict = defaultdict(list)
    
    fully_disconnected, disconnected_triple = obj.Departing()
    logger.info("Departing Disconnected Triple Done!!")
    
    for example in data:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        center_triple = (head_id, relation, tail_id)
        if center_triple in disconnected_triple:
            all_path = path_for_disconnected(data, center_triple, k_steps, num_iter)
            triple_dict[center_triple].extend(all_path)
        if center_triple not in disconnected_triple:
            all_path = obj.randomwalk(head_id, relation, tail_id, k_steps, num_iter)
            triple_dict[center_triple].extend(all_path)

    return triple_dict
"""
def process_data_chunk(chunk, obj, k_steps, num_iter, disconnected_triple, subgraph_size):
    chunk_triple_dict = defaultdict(list)

    for example in chunk:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        center_triple = (head_id, relation, tail_id)

        if center_triple in disconnected_triple:
            all_path = path_for_disconnected(chunk, center_triple, k_steps, num_iter)  # Note: pass chunk instead of the full data
        else:
            all_path = obj.randomwalk(head_id, relation, tail_id, k_steps, num_iter, subgraph_size)

        chunk_triple_dict[center_triple].extend(all_path)

    return chunk_triple_dict

def Path_Dictionary(train_path, k_steps, num_iter, obj, num_process, subgraph_size):
    data = json.load(open(train_path, 'r', encoding='utf-8'))
    triple_dict = defaultdict(list)

    fully_disconnected, disconnected_triple = obj.Departing()
    logger.info("Departing Disconnected Triple Done!!")
    logger.info("Fully Disconnected Entity: {}".format(len(fully_disconnected)))
    logger.info("Fully Disconnected Triple: {}".format(len(disconnected_triple)))
    chunk_size = len(data) // num_process
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    with multiprocessing.Pool(num_process) as pool:
        results = pool.starmap(process_data_chunk, [(chunk, obj, k_steps, num_iter, disconnected_triple, subgraph_size) for chunk in chunks])

    for chunk_result in results:
        for key, value in chunk_result.items():
            triple_dict[key].extend(value)
    # print(triple_dict[('01768969', 'derivationally related form', '02636811')])
    return triple_dict

if __name__ == "__main__":
    train_path = '/home/youminkk/Model_Experiment/2_SubGraph/1_RandomWalk_Dynamic/data/WN18RR/train.txt.json'
    obj = RandomWalk(train_path)
    path_dict = Path_Dictionary(train_path, 5, 1000,obj, 30, 200)
    keys = list(path_dict.keys())

    outliers = []

    for key in keys:
        path_lists = path_dict[key]
        x,y = len(path_lists[0]), len(path_lists[1])
        if x < 100 or y < 100:
            print(x,y)
            outliers.append(key)
    print(len(outliers))
    print(outliers)
 
