from collections import defaultdict
import networkx as nx
import pickle
import json


from logger_config import logger
from multiprocessing import Pool
import multiprocessing
import argparse
import datetime
import time
import gc
import os


def load_pkl(path:str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path:str):
    data = json.load(open(path, 'r', encoding='utf-8'))
    return data

def nxGraph(path:str):
    # We only need a undirected graph (e.g., nx.Graph())
    data = load_json(path)
    nx_G = nx.Graph()
    for ex in data:
        h, r, t = ex['head_id'], ex['relation'], ex['tail_id']
        nx_G.add_node(h)
        nx_G.add_node(t)
        nx_G.add_edge(h, t, relation=r)
    return nx_G

def get_degree_dict(nxGraph, subgraphs):
    logger.info("Stage1: Degree Weight Dictionary")
    s = time.time()
    degree_dict = defaultdict(int)
    entities = list(subgraphs.keys())
    entities = [ex[0] for ex in entities]
    for entity in entities:
        d = nxGraph.degree(entity)
        degree_dict[entity] = d
    e = time.time()
    logger.info(f"Time for building degree_dict: {datetime.timedelta(seconds = e-s)}")

    return degree_dict

def get_spw_dict(subgraph, nxGraph, num_cpu):
    logger.info("Stage2: Shortest Path Weight Dictionary")
    """
    # The key of the subgraph dictionary is the tuple of the center triple and the value is the list of subgraph triples.
    # Calculate the distance between center triple's head and other triple's tail in the subgraph.
    """
    s = time.time()
    centers = list(subgraph.keys())
    keys = list(subgraph.keys())
    logger.info("Transform!!")

    transform = defaultdict(set)
    for key in keys:
        head = key[0]
        values = subgraph[key]
        values = [ex[2] for ex in values]
        values = set(values)
        if head not in transform:
            transform[head] = set()
        transform[head] = transform[head].union(values)

    del keys

    logger.info("SPW_Dictionray!!")
    spw_dict = defaultdict(dict)
    heads = list(transform.keys())

    with Pool(num_cpu) as p:
        results = p.starmap(get_shortest_distance, [(nxGraph, head, list(transform[head])) for head in heads])

    for head, sub_dict in results:
        spw_dict[head] = sub_dict

    e = time.time()
    logger.info(f"Time for building spw_dict: {datetime.timedelta(seconds=e-s)}")
    return spw_dict

def get_shortest_distance(nxGraph, head, tail_list):
    sub_dict = {}
    for tail in tail_list:
        try:
            st = nx.shortest_path_length(nxGraph, source=head, target=tail)
            if st == 0:
                st = 1
        except nx.NetworkXNoPath:
            st = 999
        except nx.NodeNotFound:
            st = 999
        sub_dict[tail] = st
    return head, sub_dict

def main(base_dir, dataset, num_cpu, k_step, n_iter, mode, distribution):
    s = time.time()

    # Step 1) Path for Loading Data
    data_file = f"{mode}.txt.json"
    data_path = os.path.join(base_dir, dataset, data_file)

    subgraph_file = f"{mode}_{distribution}_{k_step}_{n_iter}.pkl"
    subgraph_path = os.path.join(base_dir, dataset, subgraph_file)

    # Step 2) Path for Saving Data
    # json: degree
    # pkl : shortest path, subgraphs
  
    degree_weight_file = f"Degree_{mode}.json"
    degree_weight_path = os.path.join(base_dir, dataset, degree_weight_file)

    shortest_weight_file = f"ShortestPath_{mode}.pkl"
    shortest_weight_path = os.path.join(base_dir, dataset, shortest_weight_file)

    logger.info("Start to make weigh files.")
    # Step 3) Initialization
    nx_G = nxGraph(data_path)
    subgraphs = load_pkl(subgraph_path)

    # Step 4) Degree Weight Dictionary
    degree_dict = get_degree_dict(nx_G, subgraphs)
    with open(degree_weight_path, 'w', encoding='utf-8') as f:
        json.dump(degree_dict, f)

    del degree_dict

    # Step 5) Shortest Path Length Weight Dictionary
    if mode == "train":
        shortest_dict = get_spw_dict(subgraphs, nx_G, num_cpu)
        with open(shortest_weight_path, 'wb') as file:
            pickle.dump(shortest_dict, file)
    e = time.time()
    logger.info("Done!!")
    logger.info("Total Time: {}".format(datetime.timedelta(seconds = e- s)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save path dictionary.")
    parser.add_argument("--base-dir", type=str, required=True, help="Your base directory")    
    parser.add_argument("--dataset", type=str, choices=['WN18RR', 'FB15k237', 'wiki5m_ind', 'wiki5m_trans'], required=True, help="Dataset name")
    parser.add_argument("--num-cpu", type=int, required=True, help="Number of CPUs for parallel processing")
    parser.add_argument("--k-step", type=int, required=True, help="Number of steps for the random walk")
    parser.add_argument("--n-iter", type=int, required=True, help="Number of iterations for the random walk")
    parser.add_argument("--mode", type=str, choices=['train', 'valid'], required=True, help="mode")
    parser.add_argument("--distribution", type=str, choices=["uniform", "proportional", "antithetical"], required=True, help="distribution")
    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.num_cpu, args.k_step, args.n_iter, args.mode, args.distribution)
