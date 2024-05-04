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
import numpy as np


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

def get_degree_weights(subgraph, nxGraph):
    total_dw = []
    dh_total, dt_total = [], []

    for center in subgraph:
        for triple in subgraph[center]:
            dh = nxGraph.degree(triple[0])
            dt = nxGraph.degree(triple[2])
            dh_total.extend([dh, dt])
            dt_total.extend([dt, dh])

    total_dw.extend([dh_total, dt_total])
    return total_dw

def get_spw_dict(subgraph, nxGraph, num_cpu):
    logger.info("Stage2: Shortest Path Weight Dictionary")
    s = time.time()
    centers = list(subgraph.keys())
    centers = centers[:20]
    logger.info("SPW_Dictionary!!")
    total_sw = []
    
    with Pool(num_cpu) as p:
        results = p.starmap(get_shortest_distance, [(nxGraph, center, subgraph[center]) for center in centers])

    for sub_list in results:
        total_sw.extend(sub_list)

    e = time.time()
    logger.info(f"Time for building spw_dict: {datetime.timedelta(seconds=e-s)}")
    return total_sw

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

    return sub_list

def get_final_subgraphs(subgraph):
    keys = list(subgraph.keys())
    total_subgraphs = []
    for key in keys:
        total_subgraphs.extend(subgraph[key])
    return total_subgraphs

def main(base_dir, dataset, num_cpu, k_step, n_iter, distribution):
    s = time.time()

    # Step 1) Path for Loading Data
    inpath_train = os.path.join(base_dir, dataset, 'train.txt.json')
    inpath_valid = os.path.join(base_dir, dataset, 'valid.txt.json')
    inpath_subgraphs_train = os.path.join(base_dir, dataset, f"train_{distribution}_{k_step}_{n_iter}.pkl")
    inpath_subgraphs_valid = os.path.join(base_dir, dataset, f"valid_{distribution}_{k_step}_{n_iter}.pkl")
    
    outpath_subgraphs_train = os.path.join(base_dir, dataset, f"train_{distribution}_{k_step}_{n_iter}.pkl")
    outpath_subgraphs_valid = os.path.join(base_dir, dataset, f"valid_{distribution}_{k_step}_{n_iter}.pkl")
    
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

    final_subgraphs_train = get_final_subgraphs(subgraphs_train)
    final_subgraphs_valid = get_final_subgraphs(subgraphs_valid)
    with open(outpath_subgraphs_train, 'wb') as f1:
        pickle.dump(final_subgraphs_train, f1)
        print("Done for subgraphs_train!!")
    with open(outpath_subgraphs_valid, 'wb') as f2:
        pickle.dump(final_subgraphs_valid, f2)
        print("Done for subgraphs_valid!!")
    
    del final_subgraphs_train
    del final_subgraphs_valid
    
    # Step 4) Degree Weight Dictionary
    final_degree_train = get_degree_weights(subgraphs_train, nx_G_train)
    final_degree_valid = get_degree_weights(subgraphs_valid, nx_G_valid)   
    with open(outpath_degree_train, 'wb') as f1:
        pickle.dump(final_degree_train, f1)
    with open(outpath_degree_valid, 'wb') as f2:
        pickle.dump(final_degree_valid, f2)

    del final_degree_train
    del final_degree_valid
    del subgraphs_valid
    del nx_G_valid

    # Step 5) Shortest Path Length Weight Dictionary
    final_shortest_train = get_spw_dict(subgraphs_train, nx_G_train, num_cpu)
    with open(outpath_shortest_train , 'wb') as file:
        pickle.dump(final_shortest_train, file)
     
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
