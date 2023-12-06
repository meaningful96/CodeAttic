import json
import numpy as np 
import time
import datetime
import random
from collections import defaultdict
from scipy.stats import norm
from logger_config import logger

def make_graph(train_file_path):
    graph = defaultdict(list)
    directed_graph = defaultdict(list)

    examples = json.load(open(train_file_path, 'r', encoding='utf-8'))
    train_len = len(examples)    
    for ex in examples:
        head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
        if head_id not in graph:
            graph[head_id] = list()
        graph[head_id].append([head_id, relation, tail_id])
            
        if tail_id not in graph:
            graph[tail_id] = list()
        graph[tail_id].append([tail_id, relation, head_id])

        if head_id not in directed_graph:
            directed_graph[head_id] = list()
        directed_graph[head_id].append([head_id, relation, tail_id])

    return graph, directed_graph, train_len   



class DFS_PATH:
    def __init__(self, graph, directed_graph, walks_num):
        self.walks_num = walks_num
        self.graph = graph
        self.directed_graph = directed_graph
        

    def dfs_triplet(self, start_node, walks_num):
        stack = []
        stack.append(start_node)
        seen = set()
        seen.add(start_node)
        walk = []  
        walk_nodes = []

        while len(stack) > 0:            
            vertex = stack[-1]  # Peek the top element without removing it
            unvisited_triplets = [triplet for triplet in self.graph.get(vertex, []) if triplet[2] not in seen]

            if unvisited_triplets:
                #next_triplet = unvisited_triplets[0]  # Choose the next unvisited triplet
                next_triplet = random.choice(unvisited_triplets)
                next_node = next_triplet[2]
                stack.append(next_node)
                seen.add(next_node)  
                walk_nodes.append(next_node)

                if next_triplet in self.directed_graph[vertex]:
                    walk.append(next_triplet)
                else:
                    tail, rel, head = next_triplet[0], next_triplet[1], next_triplet[2]
                    reverse = [head, rel, tail]
                    walk.append(reverse)
                    
                if len(walk) >= walks_num:
                    break
            else:
                # If there are no unvisited triplets, backtrack
                stack.pop()

        return walk, walk_nodes


    def intermediate(self):
        count = 0 
        all_pathes = []
        candidates = defaultdict(list)
        #count_dict = {}
        for node in self.graph.keys():
            dfs_path = []
            walk, walk_nodes = self.dfs_triplet(node, self.walks_num)
            candidates[node] = walk_nodes
            if len(walk) < self.walks_num:
                count +=1
                #count_dict[node] = len(walk)
                while len(walk) < self.walks_num:
                    random_entity = random.sample(self.graph.keys(), 1)[0]
                    walk2,_ = self.dfs_triplet(random_entity, self.walks_num - len(walk))
                    walk.extend(walk2)
                
            for triple in walk:
                assert len(walk) == self.walks_num
                head_id, relation, tail_id = triple[0], triple[1], triple[2]
                ex = {'head_id': head_id,'relation': relation,'tail_id': tail_id}
                dfs_path.append(ex)
                
            all_pathes.append(dfs_path)
        
        logger.info(f'cannot make {self.walks_num} path : {count}')
        return all_pathes, candidates
    

'''
start = time.time()
train_path = 'data/FB15k237/train.txt.json'
D = DFS_PATH(train_path, walks_num = 100)
print('Start dfs !!!')
all_pathes, count = D.intermediate()
print(f'cannot make 100 path : {count}')
output_path = 'data/FB15k237/train_dfs_100.json'
with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pathes, f, ensure_ascii=False, indent=4)

end = time.time()
sec = end-start
print("Taking time:", datetime.timedelta(seconds=sec))


start = time.time()
train_path = 'data/WN18RR/train.txt.json'
D = DFS_PATH(train_path, walks_num = 100)
print('Start dfs !!!')
all_pathes, count= D.intermediate()
print(f'cannot make 100 path: {count}')
output_path = 'data/WN18RR/train_dfs_100.json'
with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pathes, f, ensure_ascii=False, indent=4)

end = time.time()
sec = end-start
print("Taking time:", datetime.timedelta(seconds=sec))
'''