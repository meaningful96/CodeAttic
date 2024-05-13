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

def get_degree_weights(subgraph, nxGraph, subgraph_size):
    total_dw = defaultdict(list)

    subgraph_dict = subgraph[0]
    center_list = subgraph[1]

    for center in center_list:
        dh_sub, dt_sub = [], []
        subgraph = subgraph_dict[center]

        assert subgraph_size == len(subgraph)

        for triple in subgraph:
            dh = nxGraph.degree(triple[0])
            dt = nxGraph.degree(triple[2])
            dh_sub.extend([dh, dt])
            dt_sub.extend([dt, dh])
        assert len(dh_sub) == subgraph_size*2
        assert len(dt_sub) == subgraph_size*2
        total_dw[center].extend([dh_sub, dt_sub])
    
    return total_dw

def get_spw_dict(subgraph, nxGraph, num_cpu, subgraph_size):
    logger.info("Stage2: Shortest Path Weight Dictionary")
    s = time.time()
    subgraph_dict = subgraph[0]
    centers = subgraph[1]

    logger.info("SPW_Dictionary!!")
    total_sw = defaultdict(list)
    
    with Pool(num_cpu) as p:
        results = p.starmap(get_shortest_distance, [(nxGraph, center, subgraph_dict[center], subgraph_size) for center in centers])

    for sub_list, center in results:
        assert len(sub_list) == subgraph_size*2
        total_sw[center] = sub_list

    e = time.time()
    logger.info(f"Time for building spw_dict: {datetime.timedelta(seconds=e-s)}")
    return total_sw

def get_shortest_distance(nxGraph, center, tail_list, subgraph_size):
    sub_list = list(np.zeros(subgraph_size*2))
    
    assert len(sub_list) == subgraph_size*2
    assert len(tail_list)*2 == len(sub_list)

    head = center[0]

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
        sub_list[2*i] = 1/st
        sub_list[2*i+1] = 1/st


    return sub_list, center



def main(base_dir, dataset, num_cpu, k_step, n_iter, subgraph_size, distribution):
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
    final_degree_train = get_degree_weights(subgraphs_train, nx_G_train, subgraph_size)
    final_degree_valid = get_degree_weights(subgraphs_valid, nx_G_valid, subgraph_size)   
    with open(outpath_degree_train, 'wb') as f1:
        pickle.dump(final_degree_train, f1)
    with open(outpath_degree_valid, 'wb') as f2:
        pickle.dump(final_degree_valid, f2)

    del final_degree_train
    del final_degree_valid
    del subgraphs_valid
    del nx_G_valid

    # Step 5) Shortest Path Length Weight Dictionary
    final_shortest_train = get_spw_dict(subgraphs_train, nx_G_train, num_cpu, subgraph_size)
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
    parser.add_argument("--subgraph-size", type=int, required=True, help="Subgraph-size")
    parser.add_argument("--distribution", type=str, choices=["uniform", "proportional", "antithetical"], required=True, help="distribution")
    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.num_cpu, args.k_step, args.n_iter, args.subgraph_size, args.distribution)
