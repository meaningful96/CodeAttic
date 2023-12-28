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
    Graph, Graph_tail, diGraph = defaultdict(set), defaultdict(set), defaultdict(set)
    examples = json.load(open(train_path, 'r', encoding = 'utf-8'))
    appearance = {}

    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        appearance[(head_id, relation, tail_id)] = 0

        if head_id not in Graph:
            Graph[head_id] = set()
            Graph_tail[head_id] = set()
            diGraph[head_id] = set()
        Graph[head_id].add((head_id, relation, tail_id))
        Graph_tail[head_id].add(tail_id)
        diGraph[head_id].add((head_id, relation, tail_id))
        
        if tail_id not in Graph:
            Graph[tail_id] = set()
            Graph_tail[tail_id] = set()
        Graph[tail_id].add((tail_id, relation, head_id))
        Graph_tail[tail_id].add(head_id)    

    entities = list(Graph.keys())

    return Graph, Graph_tail, diGraph, appearance, entities

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
        self.Graph, self.Graph_tail, self.diGraph, self.appearance, self.entities = build_graph(train_path)
        logger.info("Done building Link Graph with {} nodes".format(len(self.Graph)))
        logger.info("Done building Directed Graph with {} nodes".format(len(self.diGraph)))

        self.train_dataset = json.load(open(train_path, 'r', encoding='utf-8'))
        self.degree_prob = defaultdict(list)
        self.degree_list = defaultdict(list)    

        for entity in self.entities:
            neighbors = self.get_neighbor_ent_ids(entity)
            degree_list = []
            degree_ent = []
            for neighbor in neighbors:
                degree_list.append(len(self.get_neighbor_ent_ids(neighbor)))
                degree_ent.append(neighbor)
            prob_proportional = [degree / sum(degree_list) for degree in degree_list]
            prob_antithetical = [ 1/degree  for degree in degree_list]
            prob_antithetical = [inverse_degree / sum(prob_antithetical) for inverse_degree in prob_antithetical]
            self.degree_prob[entity].append(prob_proportional)
            self.degree_prob[entity].append(prob_antithetical)
            self.degree_list[entity].extend(degree_ent)

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        neighbors = [item[2] for item in self.Graph.get(tail_id, set())]
        distant_neighbors = []
        
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))
        return list(set(neighbors + distant_neighbors))

    def get_neighbor_ent_ids(self, entity_id: str) -> List[str]:
        neighbor_ids = self.Graph_tail.get(entity_id, set())
        return sorted(list(neighbor_ids))

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
        degree_list = self.degree_list # Neighbors
        degree_prob = self.degree_prob # [0]: Degree Proportional, [1]: Degree Antithetical
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        subgraph_candidates = []         
        nh, nt = len(degree_list[head_id]), len(degree_list[tail_id])
        center_ent = [head_id, tail_id]    
        center_pro, center_ant = [nh, nt], [1/nh, 1/nt]

        prob_reset = k_steps//2 # Mean of the path length 
        r_prob = 1 / prob_reset
        s_prob = 1 - r_prob

        # Step 1. Selecting the Start Point
        
        for _ in range(num_iter):
            
            
            # Uniform Distribution
            # """
            current_entity = random.choice([head_id, tail_id])
            # """
            
            # Degree Proportional
            """
            current_entity = center_ent[weighted_random_selection(center_pro)]
            """
            
            # Degree Antithetical
            """
            current_entity = center_ent[weighted_random_selection(center_ant)]
            """

            # Step 2. Random Walk with Restart(RWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            
            triples_list = []
            triples_list.append(center_triple)
            visited = set()
            visited.add(current_entity)
            candidate = None
            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    subgraph_candidates.extend(triples_list)
                    break
                
                # Walking toward the neighbors
                if standard == 1:

                    neighbors = self.get_neighbor_ent_ids(current_entity)
                    
                    
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        # """
                        candidate = random.choice(list(neighbors))
                        # """
                        # Degree Proportional
                        """
                        candidate_prob = degree_prob[current_entity][0] # Probability
                        candidate_list = degree_list[current_entity]
                        selected_index = weighted_random_selection(candidate_prob)
                        candidate = candidate_list[selected_index]
                        """
                    
                        # Degree Antithetical
                        """
                        candidate_prob = degree_prob[current_entity][1] # Probability
                        candidate_list = degree_list[current_entity]
                        selected_index = weighted_random_selection(candidate_prob)
                        candidate = candidate_list[selected_index]
                        """   

                    visited.add(candidate)
                    if current_entity in directed_graph.keys():
                        for triple in directed_graph[current_entity]:
                            if triple[2] == candidate:
                                triples_list.append(triple)
                                append = True
                        if append == False:
                            for triple in directed_graph[candidate]:
                                if triple[2] == current_entity:
                                    triples_list.append(triple)   
                                    append = True
                
                    if current_entity not in directed_graph.keys():
                        for triple in directed_graph[candidate]:
                            if triple[2] == current_entity:
                                triples_list.append(triple)
                                append = True
                    
                    assert append == True, "No triple is appended!!"
                    current_entity = candidate                                
                
                standard = weighted_random_selection(prob_list)
            subgraph_candidates.extend(triples_list)
        subgraph_candidates = list(set(subgraph_candidates))
        return subgraph_candidates


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

    for i in range(num_iter):
        path = []
        sample = random.sample(data, k_steps)
        for ex in sample:
            path.append((ex['head_id'], ex['relation'], ex['tail_id']))
        path.append(triple)
        all_path.extend(path)
    all_path = list(set(all_path))

    return all_path


def Making_Subgraph(path_dict, candidates, subgraph_size, appearance, batch_size):
    total_subgraph = []
    batch_total, sub_total = [], []
    tmp1, tmp2 = [], []
    p = len(candidates)
    for candidate in candidates:
        subgraph = []

        # Extract the all subgraph candidates
        path_list = path_dict[candidate]    
        
        # Shuffle the candidates list
        random.shuffle(path_list)   

        # Using Set()
        hr_set = set()

        # add center triple
        subgraph.append(candidate)
        hr_set.add(candidate[0])
        hr_set.add(candidate[2])
        
        for triple in path_list:
            h, t = triple[0], triple[2]
            if h not in hr_set and t not in hr_set:
                subgraph.append(triple)
                hr_set.add(h)
                hr_set.add(t)

            if len(subgraph) >= subgraph_size:
                break

        if len(subgraph) < subgraph_size:
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1])
            num_diff = subgraph_size - len(subgraph)
            new_triples = sorted_triples[p: p + num_diff]
            triples = [item[0] for item in new_triples]
            p += num_diff
            subgraph.extend(triples)
            if p + subgraph_size >= len(appearance.keys()):
                p = 0
        
        if len(subgraph) >= subgraph_size:
            subgraph = subgraph[:subgraph_size]
        
        assert len(subgraph) == subgraph_size
        
        total_subgraph.extend(subgraph)
        
        for ex in subgraph:
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

    for example in total_subgraph:
        appearance[example] += 1

    total_subgraph = [{'head_id': head, 'relation': rel, 'tail_id': tail}
                      for head, rel, tail in total_subgraph]      

    return total_subgraph, appearance, sub_total, batch_total
   

import datetime
if __name__ == "__main__":

    train_path = '/home/youminkk/Model_Experiment/2_SubGraph/3_RWR_Dynamic/data/WN18RR/train.txt.json'
    obj = RandomWalk(train_path)
    batch_size = 1024
    subgraph = 512
    step_size = 169

    k_step = 14
    n_iter = 1000

    sd = time.time()
    path_dict = Path_Dictionary(train_path, k_step, n_iter, obj, 30, subgraph)
    ed = time.time()
    print("Time for Building Path Dictionary: {}".format(datetime.timedelta(seconds = ed - sd)))
   
    Graph, Graph_tail, diGraph, appearance, entities = build_graph(train_path)
    len_wn = len(json.load(open(train_path, 'r', encoding='utf-8')))
    num_candidates = (step_size * batch_size //(subgraph*2))
    candidates = random.sample(list(path_dict.keys()), num_candidates)
    print(num_candidates)
    sub_duplication, batch_duplication = [], []
    x, y, z1, z2 = set(), set(), set(), set()
    Epoch = 50
    for epoch in range(Epoch):
        s = time.time()
        total_subgraph, appearance_tmp, sub_total, batch_total = Making_Subgraph(path_dict, candidates, subgraph, appearance, batch_size)
        e = time.time()
        if epoch == Epoch - 1:
            print("Making Subgraph: {}".format(datetime.timedelta(seconds = e - s)))
        sub_duplication.extend(sub_total)
        batch_duplication.extend(batch_total)
        

        sorted_candidates_train = sorted(appearance_tmp.items(), key=lambda x: x[1])
        new_candidates = sorted_candidates_train[:num_candidates]
        candidates = [item[0] for item in new_candidates]
        appearance = appearance_tmp
        

        # """
        if epoch == 9:
            bc_bottom10 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom10 = [item[0] for item in bc_bottom10]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 10:
            ac_bottom11 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom11 = [item[0] for item in ac_bottom11]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom10)):
                triple1 = bc_bottom10[i]
                triple2 = ac_bottom11[i]
                x.add(triple1[0])
                x.add(triple1[2])
                y.add(triple2[0])
                y.add(triple2[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward1 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward1))
            x, y = set(), set()
            z1, z2 = set(), set()
        
        if epoch == 19:
            bc_bottom20 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom20 = [item[0] for item in bc_bottom20]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 20:
            ac_bottom21 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom21 = [item[0] for item in ac_bottom21]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom20)):
                triple1 = bc_bottom20[i]
                triple2 = ac_bottom21[i]
                x.add(triple1[0])
                x.add(triple1[2])
                y.add(triple2[0])
                y.add(triple2[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward2 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward2))
            x, y = set(), set()
            z1, z2 = set(), set()

        if epoch == 29:
            bc_bottom30 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom30 = [item[0] for item in bc_bottom30]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 30:
            ac_bottom31 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom31 = [item[0] for item in ac_bottom31]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom30)):
                triple1 = bc_bottom30[i]
                triple2 = ac_bottom31[i]
                x.add(triple1[0])
                x.add(triple1[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward3 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward3))
            x, y = set(), set()
            z1, z2 = set(), set()

        if epoch == 39:
            bc_bottom40 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom40 = [item[0] for item in bc_bottom40]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 40:
            ac_bottom41 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom41 = [item[0] for item in ac_bottom41]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom40)):
                triple1 = bc_bottom40[i]
                triple2 = ac_bottom41[i]
                x.add(triple1[0])
                x.add(triple1[2])
                y.add(triple2[0])
                y.add(triple2[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward4 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward4))
            x, y = set(), set()
            z1, z2 = set(), set()
        if epoch == 48:
            bc_bottom50 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom50 = [item[0] for item in bc_bottom50]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 49:
            ac_bottom51 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom51 = [item[0] for item in ac_bottom51]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom40)):
                triple1 = bc_bottom50[i]
                triple2 = ac_bottom51[i]
                x.add(triple1[0])
                x.add(triple1[2])
                y.add(triple2[0])
                y.add(triple2[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward5 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward5))
            x, y = set(), set()
            z1, z2 = set(), set()   
        
        """
        if epoch == 18:
            bc_bottom60 = sorted_candidates_train[num_candidates:num_candidates+3200]
            bc_bottom60 = [item[0] for item in bc_bottom60]
            print("{} Epoch Before Bottom 3200".format(epoch))
        if epoch == 19:
            ac_bottom61 = sorted_candidates_train[num_candidates:num_candidates+3200]
            ac_bottom61 = [item[0] for item in ac_bottom61]
            print("{} Epoch After Bottom 3200".format(epoch))
            for i in range(len(bc_bottom60)):
                triple1 = bc_bottom60[i]
                triple2 = ac_bottom61[i]
                x.add(triple1[0])
                x.add(triple1[2])
                y.add(triple2[0])
                y.add(triple2[2])
                z1.add(triple1)
                z2.add(triple2)
            triple_backward6 = len(z2- z1)
            print("min-Max Triple Duplication: {}".format(triple_backward6))
            x, y = set(), set()
            z1, z2 = set(), set() 
        """
        # """

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
        if value >= mean_appearance:
            cnt += 1
    
    print("Couting zero appearance: {}".format(count))
    print("Mean Duplication_Batch: {}".format(batch_meanDuple))
    print("Mean Duplication_Subgraph: {}".format(sub_meanDuple))
    print("Mean appearance: {}".format(mean_appearance))
    print("More than average triples: {}".format(cnt))
 
