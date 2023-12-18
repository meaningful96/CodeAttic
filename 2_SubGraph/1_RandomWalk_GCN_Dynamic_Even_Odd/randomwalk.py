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

def weighted_random_selection(prob_distribution):
    # Generating a list of indices based on the probability distribution
    indices = list(range(len(prob_distribution)))
    
    # Performing weighted random selection for a single trial
    selected_index = random.choices(indices, weights=prob_distribution, k=1)[0]

    return selected_index

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

    def randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int) -> List[list]:
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
                # Unifrom Distribution
                candidate = random.choice(list(neighbors)) 

                # Degree Proportional
                """
                degree_list = []
                for neighbor in neighbors:
                    degree_list.append(len(self.get_neighbor_ids(neighbor, 1)))
                prob_list = [degree / sum(degree_list) for degree in degree_list]
                selected_index = weighted_random_selection(prob_list)
                candidate = neighbors[selected_index]
                """

                # Degree Antithetical
                """
                # degree_list = []
                # for neighbor in neighbors:
                #     prob = 1 / len(self.get_neighbor_ids(neighbor, 1))
                #     degree_list.append(prob)
                # prob_list = [inverse_degree / sum(degree_list) for inverse_degree in degree_list]
                # selected_index = weighted_random_selection(prob_list)
                # candidate = neighbors[selected_index]
                """

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

            # remove the duplication triples in 'path'
            path = list(set(path))
            all_path_head.append(path)  
        
        # remove the duplication head paths and put them into the total_path    
        all_path_head_tuples = [tuple(map(tuple, path_list)) for path_list in all_path_head]
        unique_head_tuples = set(all_path_head_tuples)
        unique_head_paths = [list(map(tuple, path_tuple)) for path_tuple in unique_head_tuples]
        total_path.append(unique_head_paths)

        all_path_tail = []
        ## for 'tail' with center triple
        for _ in range(num_iter // 2):
            path = []
            current_entity = tail_id
            path.append(center_triple)
           
            for _ in range(k_steps):
                append = False
                neighbors = self.get_neighbor_ids(current_entity, 1)
                # Unifrom Distribution
                candidate = random.choice(list(neighbors)) 

                # Degree Proportional
                """
                degree_list = []
                for neighbor in neighbors:
                    degree_list.append(len(self.get_neighbor_ids(neighbor, 1)))
                prob_list = [degree / sum(degree_list) for degree in degree_list]
                selected_index = weighted_random_selection(prob_list)
                candidate = neighbors[selected_index]
                """

                # Degree Antithetical
                """
                # degree_list = []
                # for neighbor in neighbors:
                #     prob = 1 / len(self.get_neighbor_ids(neighbor, 1))
                #     degree_list.append(prob)
                # prob_list = [inverse_degree / sum(degree_list) for inverse_degree in degree_list]
                # selected_index = weighted_random_selection(prob_list)
                # candidate = neighbors[selected_index]
                """

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

            # remove the duplication triples in 'path'
            path = list(set(path))
            all_path_tail.append(path)
        
        # remove the duplication tail paths and put them into the total_path
        all_path_tail_tuples = [tuple(map(tuple, path_list)) for path_list in all_path_tail]
        unique_tail_tuples = set(all_path_tail_tuples)
        unique_tail_paths = [list(map(tuple, path_tuple)) for path_tuple in unique_tail_tuples]
        total_path.append(unique_tail_paths)
        return total_path

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
        results = pool.starmap(process_data_chunk, [(chunk, data, obj, k_steps, num_iter, disconnected_triple) for chunk in chunks])

    for chunk_result in results:
        for key, value in chunk_result.items():
            triple_dict[key].extend(value)
    return triple_dict

def process_data_chunk(chunk, data, obj, k_steps, num_iter, disconnected_triple):
    chunk_triple_dict = defaultdict(list)

    for example in chunk:
        head_id, relation, tail_id = example['head_id'], example['relation'], example['tail_id']
        center_triple = (head_id, relation, tail_id)

        if center_triple in disconnected_triple:
            all_path = path_for_disconnected(data, center_triple, k_steps, num_iter)  # Note: pass chunk instead of the full data
        else:
            all_path = obj.randomwalk(head_id, relation, tail_id, k_steps, num_iter)

        chunk_triple_dict[center_triple].extend(all_path)

    return chunk_triple_dict

def path_for_disconnected(data, triple, k_steps, num_iter):
    all_path = []

    ## for head
    path_head = []
    for i in range(num_iter//2):
        path = []
        sample = random.sample(data, k_steps)
        for ex in sample:
            path.append((ex['head_id'], ex['relation'], ex['tail_id']))
        path.append(triple)
        path_head.append(tuple(path))
    all_path_head = list(set(path_head))

    ## for tail
    path_tail = []
    for i in range(num_iter//2):
        path = []
        sample = random.sample(data, k_steps)
        for ex in sample:
            path.append((ex['head_id'], ex['relation'], ex['tail_id']))
        path.append(triple)
        path_tail.append(tuple(path))
    all_path_tail = list(set(path_tail))    

    all_path.append(all_path_head)
    all_path.append(all_path_tail)

    return all_path

def Making_Subgraph(path_dict, candidates, subgraph_size, appearance, batch_size):
    total_subgraph = []
    batch_total, sub_total = [], []
    tmp1, tmp2 = [], []
    tmp3, tmp4 = [], [] 

    batch1, batch2 = [], []
    p1 = len(candidates)
    p2 = len(candidates)
    for candidate in candidates:
        subgraph = []
        # Extract the head paths and tail paths
        path_list = path_dict[candidate]
        head_path_list = path_list[0]
        tail_path_list = path_list[1]       

        # Shuffle the each path list
        random.shuffle(head_path_list)
        random.shuffle(tail_path_list)
        
        n = min(len(head_path_list), len(tail_path_list))

        # even -> sub_batch1
        # odd  -> sub_batch2
        sub_batch1, sub_batch2 = [], []
        sub_batch1.append(candidate)
        sub_batch2.append(candidate)
        for i in range(n):
            head_candidates, tail_candidates = head_path_list[i], tail_path_list[i]
            sub_batch1.extend(head_candidates[::2])
            sub_batch1.extend(tail_candidates[::2])
            sub_batch1 = list(set(sub_batch1))

            if len(sub_batch1) >= subgraph_size:
                break

        if len(sub_batch1) >= subgraph_size:
            sub_batch1 = sub_batch1[:subgraph_size]

        if len(sub_batch1) < subgraph_size:
            # If the subgraph size is less than the given size
            # add the triples that are sparsely appearaed
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1]) # sorted by values
            num_diff = subgraph_size - len(sub_batch1)
            new_triples = sorted_triples[p1:p1+num_diff]
            triples = [item[0] for item in new_triples] # item[0] = key = Triple
            p1 += num_diff
            sub_batch1.extend(triples)

        assert len(sub_batch1) == subgraph_size

        for i in range(n):
            head_candidates, tail_candidates = head_path_list[i], tail_path_list[i]
            sub_batch2.extend(head_candidates[1::2])
            sub_batch2.extend(tail_candidates[1::2])
            sub_batch2 = list(set(sub_batch2))

            if len(sub_batch2) >= subgraph_size:
                break

        if len(sub_batch2) >= subgraph_size:
            sub_batch2 = sub_batch2[:subgraph_size]

        if len(sub_batch2) < subgraph_size:
            # If the subgraph size is less than the given size
            # add the triples that are sparsely appearaed
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1]) # sorted by values
            num_diff = subgraph_size - len(sub_batch2)
            new_triples = sorted_triples[p2:p2+num_diff]
            triples2 = [item[0] for item in new_triples] # item[0] = key = Triple
            p2 += num_diff
            sub_batch2.extend(triples2)

        assert len(sub_batch2) == subgraph_size

        batch1.extend(sub_batch1)
        batch2.extend(sub_batch2)
        
        for ex in sub_batch1:
            tmp1.append(ex[0])
            tmp1.append(ex[2])
            
        x1 = subgraph_size*2 - len(list(set(tmp1)))    
        sub_total.append(x1)
        tmp2.extend(tmp1)
        tmp1 = []
        if len(tmp2) == batch_size:
            y1 = batch_size - len(list(set(tmp2)))
            batch_total.append(y1)
            tmp2 = []
        
        for ex in sub_batch2:
            tmp3.append(ex[0])
            tmp3.append(ex[2])          
        
        x2 = subgraph_size*2 - len(list(set(tmp3)))    
        sub_total.append(x2)
        tmp4.extend(tmp3)
        tmp3 = []
        if len(tmp4) == batch_size:
            y2 = batch_size - len(list(set(tmp4)))
            batch_total.append(y2)
            tmp4 = []
      
    total_subgraph.extend(batch1)
    total_subgraph.extend(batch2)

    for example in total_subgraph:
        appearance[example] += 1

    total_subgraph = [{'head_id': head, 'relation': rel, 'tail_id': tail}
                      for head, rel, tail in total_subgraph]      

    return total_subgraph, appearance, sub_total, batch_total
    

import time
import datetime
if __name__ == "__main__":

    train_path = '/home/youminkk/Model_Experiment/2_SubGraph/1_RandomWalk_GCN_Dynamic/data/WN18RR/valid.txt.json'
    obj = RandomWalk(train_path)
    batch_size = 1024
    subgraph = 4
    step_size = 10
    path_dict = Path_Dictionary(train_path, 5, 10, obj, 30, subgraph)
    keys = list(path_dict.keys())
    outliers = []
    for key in keys:
        path_lists = path_dict[key]
        x,y = len(path_lists[0]), len(path_lists[1])
        if x < 10 or y < 10:
            # print(x,y)
            outliers.append(key)
    print(len(outliers))
    # print(outliers)

    Graph, diGraph, appearance, entities = build_graph(train_path)
    len_wn = len(json.load(open(train_path, 'r', encoding='utf-8')))
    num_candidates = (step_size * batch_size //subgraph)
    candidates = random.sample(list(path_dict.keys()), num_candidates)

    sub_duplication, batch_duplication = [], []

    for epoch in range(50):
        s = time.time()
        total_subgraph, appearance_tmp, sub_total, batch_total = Making_Subgraph(path_dict, candidates, subgraph, appearance, batch_size)
        e = time.time()
        if epoch == 49:
            print("Making Subgraph: {}".format(datetime.timedelta(seconds = e - s)))
        sub_duplication.extend(sub_total)
        batch_duplication.extend(batch_total)

        sorted_candidates_train = sorted(appearance_tmp.items(), key=lambda x: x[1])
        new_candidates = sorted_candidates_train[:num_candidates]
        candidates = [item[0] for item in new_candidates]
        appearance = appearance_tmp
    
    batch_meanDuple = sum(batch_duplication) / len(batch_duplication)
    sub_meanDuple = sum(sub_duplication) / len(sub_duplication)

    appearance_value = list(appearance.values())
    count = 0
    mean_appearance = sum(appearance_value) / len(appearance_value)
    for value in appearance_value:
        if value == 0:
            count += 1

    cnt = 0
    for value in appearance_value:
        if value <= mean_appearance:
            cnt += 1
    
    print("Couting zero appearance: {}".format(count))
    print("Mean Duplication_Batch: {}".format(batch_meanDuple))
    print("Mean Duplication_Subgraph: {}".format(sub_meanDuple))
    print("Mean appearance: {}".format(mean_appearance))
    print("More than average entities: {}".format(cnt))

 
