import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


# Step 1) DataLoad

#-- FB15k-237, WN18RR ------------------------------------------------------------#

train_wn = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/wn18rr/train.tsv", sep = "\s+")
test_wn = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/wn18rr/test.tsv", sep = "\s+")

train_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/train.tsv", sep = "\s+")
test_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/test.tsv", sep = "\s+")

train_wn, test_wn = np.array(train_wn), np.array(test_wn)
train_fb, test_fb = np.array(train_fb), np.array(test_fb)

#-- UMLS ------------------------------------------------------------#

train_umls = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/umls/train.tsv", sep = "\s+")
test_umls = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/umls/test.tsv", sep = "\s+")

train_umls, test_umls = np.array(train_umls), np.array(test_umls)


#--------------------------------------------------------------------#

# Step 2) Triple Detaching
## wn18rr
head_wn = train_wn[:,0]
relation_wn = train_wn[:,1]
tail_wn = train_wn[:,2]

## fb15k-237
head_fb = train_fb[:,0]
relation_fb = train_fb[:,1]
tail_fb = train_fb[:,2]

## UMLS

head_umls = train_umls[:,0]
relation_umls = train_umls[:,1]
tail_umls = train_umls[:,2]

#------------------------------------------------------------------#
# Step 3) Constructing the Graph

## wn18rr
G_wn = nx.Graph() #nx.DiGraph()면 directed Graph가 생성됨.

for head, relation, tail in zip(head_wn, relation_wn, tail_wn):
    G_wn.add_edge(head, tail, relation=relation)

print("Total relation number of WN18RR:",len(G_wn.edges))
print("Varience of relation in WN18RR:", len(relation_wn) - len(G_wn.edges))
print("----------------------------------------------------")
## fb15k-237
G_fb = nx.Graph()

for head, relation, tail in zip(head_fb, relation_fb, tail_fb):
    G_fb.add_edge(head, tail, relation=relation)

print("Total relation number of FB15K-237:",len(G_fb.edges))
print("Varience of relation in FB15K-237:", len(relation_fb) - len(G_fb.edges))
print("----------------------------------------------------")
#------------------------------------------------------------------#

## umls

G_umls = nx.Graph()

for head, relation, tail in zip(head_umls, relation_umls, tail_umls):
    G_umls.add_edge(head, tail, relation=relation)

print("Total relation number of UMLS:",len(G_umls.edges))
print("Varience of relation in UMLS:", len(relation_umls) - len(G_umls.edges))
print("----------------------------------------------------")


#%%

def BFS(Graph, Start_node, End_node):
    visited = set() # 방문한 노드를 저장할 빈 set을 만듬
    queue = deque([Start_node]) # 새로운 quere를 만들고 Start_node를 Enqueue해 추가함.
    distance = {Start_node : 0} # Start node에서 visited 사이에 저장된 노드들의 거리
                                # print(len(distance)) 하면 거리 출력
    shortest_path = []
    print("Initial Values!!!!")
    print(visited)
    print(queue)
    print(len(distance))
    print("-"*60)           
    print("Algorithm Start!!!!")
    
    while queue: # queue에 노드가 있으면 계속 진행 
        node = queue.popleft() # queue에 있는 노드 삭제
        
        if node == End_node: # 만약 삭제된 노드가 End_node이면 종료 
            return distance[node]
        
        if node not in visited: # 만약 노드를 아직 방문하지 않았다면
            visited.add(node) # visited에 마킹
            
            for neighbor in Graph.neighbors(node): # Go through all neighbors of this node
                if neighbor not in visited and neighbor not in queue: # 만약 방문하지 않은 이웃이 있다면
                    queue.append(neighbor) # queue에 이웃 추가
                    distance[neighbor] = distance[node] + 1 # 시작노드에서 이웃 노드로의 거리 계산
                    print(distance)
                    print("Path Length is:", max(distance.values()))
                shortest_path.append(max(distance.values()))
        shortest_path = list(set(shortest_path))
    
        print("="*29, "Result", "="*29)
        print("Shortest Path Length is: ", shortest_path[0])
        print("Maximum Path Length is: " ,shortest_path[len(shortest_path) - 1])
        
    
    # If the end node is not reachable from the start node, return None
    return None       
