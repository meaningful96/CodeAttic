"""
Created on meaningful96

DL Project
"""

import json
from typing import List, Dict, Tuple
from collections import defaultdict, deque
import random
import time
import datetime

class LinkGraph:
    def __init__(self, train_path: str):
        print('Start to build link graph!!!')
        # id -> {(relation, id)}
        self.graph = defaultdict(set)
        
        # Directed Graph
        self.directed_graph = defaultdict(set)
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        self.head = []
        self.tail = []

        for ex in examples:
            head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
            
            # Add to graph with all details
            if head_id not in self.graph:
                self.graph[head_id] = set()
                self.directed_graph[head_id] = set()
            # Link Graph(Undirected Graph)
            self.graph[head_id].add((head_id, relation, tail_id)) 
            # Extraction Graph(Directed Graph) 
            self.directed_graph[head_id].add((head_id, relation, tail_id))

            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add((tail_id, relation, head_id))           
        
        print('Done building link graph with {} nodes'.format(len(self.graph)))
        print('Done building DIRECTED graph with {} nodes'.format(len(self.directed_graph)))
        

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        # Fetch immediate neighbors for the given tail_id
        neighbors = [item[2] for item in self.graph.get(tail_id, set())]

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
                for neighbor in self.graph.get(node, []): # Link graph
                    queue.append((neighbor[2], depth + 1))
        
        return visited
        
    def create_positive_samples(self, head_id: str, tail_id: str, tail_hops: int, max_samples: int) -> List[Dict[str, any]]:
        link_graph = self.graph
        directed_graph = self.directed_graph
        check_dict = {}

        # Exclude tail entity and its neighbors determined by tail_hops.
        exclude_ids = self.get_neighbor_ids(tail_id, tail_hops)
        exclude_ids.append(tail_id)
        exclude_ids.append(head_id)
    
        # Calculate the hop distance for all entities starting from the head entity.
        hop_distances = self.bfs(head_id)    
    
        # Sort entities based on their proximity to the head entity.
        sorted_entities = sorted(hop_distances.keys(), key=lambda k: hop_distances[k])
        results = []
        sample_count = 0
        
        for entity in sorted_entities:
            # Stop if we've collected the desired number of samples.
            if sample_count >= max_samples:
                break
            # Retrieve triples that are associated with the current entity.
            for triple in directed_graph.get(entity, []):
                # Add the triple to the sample list if the tail is not in the exclude_ids.
                if triple[0] not in exclude_ids and triple[2] not in exclude_ids:
                    results.append({
                        "head_id": triple[0],
                        "relation": triple[1],
                        "tail_id": triple[2],
                     })
                    sample_count += 1
                    check_dict[triple[0]] = set()
                    check_dict[triple[2]] = set()
                    check_dict[triple[0]].add(triple[2])
                    check_dict[triple[2]].add(triple[0])
                    if sample_count >= max_samples:
                        break
         
        #print("check_dict", len(check_dict)) 
        
        # Logic to add random samples for disconnected entities
        all_entities = list(self.directed_graph.keys())
        
        # print("All_entities",len(all_entities))
        # print(len(set(check_dict.keys())))
        # breakpoint()
        
        unvisited_entities = set(all_entities) - set(check_dict.keys())
        
        # print("unvisited_entities",len(unvisited_entities))
        # breakpoint()
        
        # Set for tracking existing samples to prevent duplication
        existing_samples = {(sample['head_id'], sample['tail_id']) for sample in results}
        
        while sample_count < max_samples and unvisited_entities:
            random_entity = random.choice(list(unvisited_entities))
            random_sample = random.choice(list(directed_graph[random_entity]))

            # Retry sampling if the conditions are not satisfied
            if (
                random_sample[0] not in exclude_ids 
                and random_sample[2] not in exclude_ids 
                and (random_sample[0], random_sample[2]) not in existing_samples  # Ensure sample uniqueness
            ):
                results.append({
                    "head_id": random_sample[0],
                    "relation": random_sample[1],
                    "tail_id": random_sample[2],
                })
                sample_count += 1
                # Add the newly added sample to existing_samples
                existing_samples.add((random_sample[0], random_sample[2]))
            if sample_count == max_samples:
                break

            # Remove the entity from unvisited_entities regardless of whether a sample was added
            if len(unvisited_entities) > 1:
                unvisited_entities.remove(random_entity)
        
        return results


import multiprocessing

def create_positive_samples_for_example(example, G):
    head_id = example['head_id']
    tail_id = example['tail_id']
    positive_samples = G.create_positive_samples(head_id, tail_id, 1, 511)
    positive_samples.insert(0, example)
    return positive_samples

if __name__ == "__main__":
    start_time = time.time()
    train_path_fb = "/home/youminkk/SimKGC/data/FB15k237/train.txt.json"
    with open(train_path_fb, 'r', encoding='utf-8') as f:
        train_data_wn = json.load(f)

    G_fb = LinkGraph(train_path_fb)

    num_processes = 40  # 사용할 CPU 코어 수
    pool = multiprocessing.Pool(processes=num_processes)

    # pool.map 함수에 create_positive_samples_for_example 함수와 인자를 제공
    all_positive_samples = pool.starmap(create_positive_samples_for_example, [(example, G_fb) for example in train_data_wn])

    pool.close()
    pool.join()

    # 결과를 JSON 파일로 저장
    output_path = "/home/youminkk/SimKGC/data/FB15k237/Samples_without_duplication_train_512.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_positive_samples, f, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    sec = end_time - start_time
    time = datetime.timedelta(seconds = sec)
    print("Total Length:", len(all_positive_samples))
    print("Positive samples of WN18RR saved")
    print("Done!!")
    print("Total Taking Time: {}".format(time))



