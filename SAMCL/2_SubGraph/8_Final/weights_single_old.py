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
    
    logger.info("SPW_Dictionray!!")
    spw_dict = defaultdict(list)
    
    with Pool(num_cpu) as p:
        results = p.starmap(get_shortest_distance, [(nxGraph, center, subgraph[center]) for center in centers])

    for head, sub_list in results:
        spw_dict[head] = sub_list

    e = time.time()
    logger.info(f"Time for building spw_dict: {datetime.timedelta(seconds=e-s)}")
    return spw_dict

import numpy as np
def get_shortest_distance(nxGraph, center, tail_list):
    sub_list = list(np.zeros(len(tail_list)*2))
    head = center[0]
 
    assert len(sub_list) == 1024

    for i, triple in enumerate(tail_list):
        tail = triple[2]
        try:
            st = nx.shortest_path_length(nxGraph, source=head, target=tail)
            if st == 0:
                st = 1
        except nx.NetworkXNoPath:
            st = 999
        except nx.NodeNotFound:
            st = 999
        sub_list[2*i] = st
        sub_list[2*i+1] = st

    return center, sub_list

def get_degree_weights(subgraph, nxGraph):
   degree_dict = defaultdict(dict)
   logger.info("Degree Dictionary!!")
   
   for center in subgraph:
       dh_list, dt_list = get_degree(nxGraph, center, subgraph[center])
       degree_dict[center]['dh'] = dh_list
       degree_dict[center]['dt'] = dt_list
   
   return degree_dict

def get_degree(nxGraph, center, tail_list):
   dh_list = np.zeros(len(tail_list)*2)
   dt_list = np.zeros(len(tail_list)*2)
   
   for i, triple in enumerate(tail_list):
       dh = nxGraph.degree(triple[0])
       dt = nxGraph.degree(triple[2])
       dh_list[2*i] = dh
       dh_list[2*i+1] = dt
       dt_list[2*i] = dt
       dt_list[2*i+1] = dh
   
   return dh_list, dt_list




def main(base_dir, dataset, num_cpu, k_step, n_iter, distribution):
    s = time.time()

    # Step 1) Path for Loading Data
    inpath_train = os.path.join(base_dir, dataset, 'train.txt.json')
    inpath_valid = os.path.join(base_dir, dataset, 'valid.txt.json')
    inpath_subgraphs_train = os.path.join(base_dir, dataset, f"train_{distribution}_{k_step}_{n_iter}.pkl")
    inpath_subgraphs_valid = os.path.join(base_dir, dataset, f"valid_{distribution}_{k_step}_{n_iter}.pkl")

    outpath_degree_train = os.path.join(base_dir, dataset, "Degree_train.pkl")
    outpath_degree_valid = os.path.join(base_dir, dataset, "Degree_valid.pkl")
    outpath_shortest_train = os.path.join(base_dir, dataset, "ShortestPath_train.pkl")

    # Step 2) Initialization
    logger.info("Build NetworkX Graph!!")
    nx_G_train = nxGraph(inpath_train)
    nx_G_valid = nxGraph(inpath_valid)

    with open(inpath_subgraphs_train, 'rb') as f:
        subgraphs_train = pickle.load(f)
    with open(inpath_subgraphs_valid, 'rb') as f:
        subgraphs_valid = pickle.load(f)

    # Step 4) Degree Weight Dictionary
    # degree_dict = get_degree_dict(nx_G, subgraphs)
    degree_dict_train = get_degree_weights(subgraphs_train, nx_G_train)
    degree_dict_valid = get_degree_weights(subgraphs_valid, nx_G_valid)
    with open(outpath_degree_train, 'wb') as f1:
        pickle.dump(degree_dict_train, f1)
    with open(outpath_degree_valid, 'wb') as f2:
        pickle.dump(degree_dict_valid, f2)

    del degree_dict_train
    del degree_dict_valid
    del subgraphs_valid
    del nx_G_valid

    # Step 5) Shortest Path Length Weight Dictionary
    shortest_dict = get_spw_dict(subgraphs_train, nx_G_train, num_cpu)
    with open(outpath_shortest_train , 'wb') as file:
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
    parser.add_argument("--distribution", type=str, choices=["uniform", "proportional", "antithetical"], required=True, help="distribution")
    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.num_cpu, args.k_step, args.n_iter, args.distribution)
