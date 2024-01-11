import torch
import json
import numpy as np 
import time
import datetime
from collections import defaultdict
from scipy.stats import norm
import random
import networkx as nx

import multiprocessing as mp
from multiprocessing import Pool

from config import args
from triplet import LinkGraph
from dfs import DFS_PATH
from doc import Example, collate

def node_degree(graph):
    #make uniform dist 
    node_deg = dict()
    item_pop = list()
    deg_sum = len(graph)
    
    for key in graph.keys():
        node_deg[key] = 1/deg_sum
    return node_deg


def negative_sampling(model, hr_vector, candidates, q1_dict, walks_num, N_steps, N_negs, graph, directed_graph, G=None, head_ids=None):    
   
    #####################################
    # In MCNS paper, they explained 
    # q(y|x) is defined by mixing uniform sampling 
    # and sampling from the nearest 𝑘 nodes with 1/2 probability each.
    # candidates : dfs path
    # n_steps = 10.. burn in period
    # output -> sampled_negatives : negatives entity id, negative_vector: negatives embedding 
    #####################################
   
    distribution = norm.pdf(np.arange(0,walks_num,1), 50, 10)   
    distribution = [i/np.sum(distribution) for i in distribution ]
    
    cur_state = None
    count = 0
    sampled = 0   
    sampled_negatives = list()

    for i in range(len(hr_vector)):
        while sampled < N_negs:
            #initialize current negative node at random 
            if cur_state is None:
                cur_state = random.choice(list(q1_dict.keys()))
    
            count +=1
            sample_num = np.random.random() 
            #Sampling from uniform dist 
            if sample_num < 0.5:
                #sample a node y from q(y|x)
                y = np.random.choice(list(q1_dict.keys()), 1, p = list(q1_dict.values()))[0]
                #q(y|x) 
                q_prob = q1_dict[y] 
                #q(x|y) 
                q_prob_next = q1_dict[cur_state] 

            #Sampling from the nearest k nodes 
            else:
                if len(candidates[cur_state]) == walks_num:
                    y = np.random.choice(candidates[cur_state], 1, p=distribution)[0]
                    index = candidates[cur_state].index(y)
                    #q(y|x)
                    q_prob = distribution[index]
                    node_list_next = candidates[y[2]]
                
                    if cur_state in node_list_next:
                        index_next = node_list_next.index(cur_state)
                        q_prob_next = distribution[index_next]
                    else:
                        q_prob_next = q1_dict[cur_state]
            
                else:
                    y = np.random.choice(list(q1_dict.keys()),1, p=list(q1_dict.values()))[0]
                    q_prob = q1_dict[y]
                    q_prob_next = q1_dict[cur_state]
            
            if count > N_steps:
                sampled_negatives.append(cur_state)
                cur_state = y
                sampled +=1 
            else: 
                #Generate a uniform random number r 
                u = np.random.rand()
                
                y_vector_list = []
                cur_vector_list = []
            
                y_vector = Example(head_id='', relation = '', tail_id = y)
                cur_vector = Example(head_id ='', relation = '', tail_id = cur_state)
                
                y_vector_list.append(y_vector.vectorize())
                cur_vector_list.append(cur_vector.vectorize())
                
                y_vector = collate(y_vector_list)
                cur_vector = collate(cur_vector_list)
                
                y_vector['only_ent_embedding'] = True 
                cur_vector['only_ent_embedding'] = True 
                y_emb = model(**y_vector)['ent_vectors']
                cur_emb = model(**cur_vector)['ent_vectors']
                
                #alpha = 0.25
                p_prob = (torch.unsqueeze(hr_vector[i],0).mm(y_emb.t())) ** 0.25 
                p_prob_next = (torch.unsqueeze(hr_vector[i],0).mm(cur_emb.t())) ** 0.25
                
                #calculate acceptance ratio
                A_a = (p_prob * q_prob_next) / (p_prob_next * q_prob)
                next_state = list()
              
                for i in list(range(len(cur_state))):
                    alpha = min(1, A_a)
                    #accept
                    if u < alpha:
                        next_state = y
                    #reject
                    else:
                        next_state = cur_state
                cur_state = next_state       
        #initilize num_sampled
        sampled = 0
    sampled_negative_triplets = []
    
    
    for neg in sampled_negatives:
        neg_triplet = random.choice(graph[neg])
        if neg_triplet not in directed_graph[neg]:
            head_id, relation, tail_id = neg_triplet[2], neg_triplet[1], neg_triplet[0]
        else:
            head_id, relation, tail_id = neg_triplet[0], neg_triplet[1], neg_triplet[2]
        
        ex = {'head_id': head_id,'relation':relation,'tail_id':tail_id}
        sampled_negative_triplets.append(ex)
    
    #for checking shortest path
    if G and head_ids:
        shortest_path_len = []
        for i in range(len(sampled_negatives)):
            try:
                shortest_path = nx.shortest_path_length(G, source=head_ids[i//N_negs], target=sampled_negatives[i])
            except nx.NetworkXNoPath:
                shortest_path = 0
            shortest_path_len.append(shortest_path)
    
        return sampled_negative_triplets, shortest_path_len

    return sampled_negative_triplets
