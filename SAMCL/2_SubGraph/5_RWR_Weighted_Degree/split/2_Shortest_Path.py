import networkx as nx
import json
import datetime
import time
import pickle
from collections import defaultdict
from multiprocessing import Pool
import argparse
import os

def calculate_shortest_paths(args):
    source, all_entities2, graph = args
    sub_dict = {}
    for target in all_entities2:
        try:
            st = nx.shortest_path_length(graph, source=source, target=target)
            if st == 0:
                st = 1
        except nx.NetworkXNoPath:
            st = len(all_entities2)
        sub_dict[target] = st
    return source, sub_dict

def main(dataset_name):
    base_dir = "/Hard_Negative/data"
    input_path = os.path.join(base_dir, dataset_name, "valid.txt.json")
    output_path = os.path.join(base_dir, dataset_name, "train_st.pkl")

    G = nx.Graph()
    data = json.load(open(input_path, 'r', encoding='utf-8'))

    all_entities = set()
    for item in data:
        G.add_node(item['head_id'], label=item['head'])
        G.add_node(item['tail_id'], label=item['tail'])
        G.add_edge(item['head_id'], item['tail_id'], relation=item['relation'])
        all_entities.add(item['head_id'])
        all_entities.add(item['tail_id'])

    all_entities_list = list(all_entities)
    st_dict = defaultdict(dict)

    print("Start to Build Shortest Path Dictionary!!")
    start_time = time.time()

    with Pool(processes=40) as pool:  
        results = pool.map(calculate_shortest_paths, [(entity, all_entities_list, G) for entity in all_entities_list])

    for source, sub_dict in results:
        st_dict[source] = sub_dict

    end_time = time.time()
    print("Time for ST Dict: {}".format(datetime.timedelta(seconds=end_time - start_time)))

    print("Storing the Dictionary!!")
    storage_start = time.time()
    with open(output_path, 'wb') as f:
        pickle.dump(st_dict, f)
    storage_end = time.time()
    print("Time for Storing: {}".format(datetime.timedelta(seconds=storage_end - storage_start)))
    print("Done!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate shortest paths in a graph.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., WN18RR, FB15k237)")

    args = parser.parse_args()

    main(args.dataset)
