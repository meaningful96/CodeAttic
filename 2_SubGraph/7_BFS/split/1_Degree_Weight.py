import os
import json
import argparse
import time
import datetime
from typing import List, Dict
from collections import defaultdict

# Assuming LinkGraph is a defined class or function in your original code
# from your_module import LinkGraph

class LinkGraph:
    def __init__(self, train_path:str):
        self.Graph, _ = linkGraph(train_path)

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[int]: # mapping: tail_id:str, List[str] -> tail_id:int, List[int]
        if n_hops <= 0:
            return []

        neighbors = [item[2]for item in self.Graph.get(tail_id, set())]
        distant_neighbors = []
        
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))
        return list(set(neighbors + distant_neighbors)) 

def linkGraph(train_path:str):
    Graph, Graph_tail, diGraph = defaultdict(set), defaultdict(set), defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding = 'utf-8'))
    appearance = {}

    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0

        if head_id not in Graph:
            Graph[head_id] = set()
        Graph[head_id].add((head_id, relation, tail_id))
        
        if tail_id not in Graph:
            Graph[tail_id] = set()
        Graph[tail_id].add((tail_id, relation, head_id))    

    entities = list(Graph.keys())

    return Graph, entities

    def create_file_path(base_dir, task_name, file_name):
        return os.path.join(base_dir, task_name, file_name)

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Name of the task (e.g., WN18RR)")
    args = parser.parse_args()

    base_dir = "/home/youminkk/Model_Experiment/2_SubGraph/5_RWR_weighted_DEGREE/data"
    file_name = "degree_weight.json"

    # Directory for the task
    task_dir = os.path.join(base_dir, args.task)

    # Ensuring the task directory exists
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    # Paths for train and valid files
    train_path = os.path.join(task_dir, 'train.txt.json')
    valid_path = os.path.join(task_dir, 'valid.txt.json')

    # Your existing logic to build the link graph and calculate degree
    print("Start to build a Link Graph")
    linkGraph_train = LinkGraph(train_path)
    linkGraph_valid = LinkGraph(valid_path)

    _, ent_train = linkGraph(train_path)
    _, ent_valid = linkGraph(valid_path)

    degree_train, degree_valid = defaultdict(int), defaultdict(int)

    print("Degree Sampling start !!!!")
    x = []
    s = time.time()
    for entity in ent_train:
        d = len(linkGraph_train.get_neighbor_ids(entity, 1))
        degree_train[entity] = d
        x.append(d)

    average_degree = sum(x) / len(x)
    print(average_degree)

    for entity in ent_valid:
        d = len(linkGraph_valid.get_neighbor_ids(entity, 1))
        degree_valid[entity] = d
    e = time.time()    
    print('Done!!!')
    print("Time for Degree Dictionary: {}".format(datetime.timedelta(seconds=e-s)))

    # Saving the data
    out_train = os.path.join(task_dir, 'degree_train.json')
    out_valid = os.path.join(task_dir, 'degree_valid.json')

    with open(out_train, 'w', encoding='utf-8') as f:
        json.dump(degree_train, f)

    with open(out_valid, 'w', encoding='utf-8') as f:
        json.dump(degree_valid, f)

    print(f"Data saved to {out_train} and {out_valid}")

if __name__ == "__main__":
    main()
