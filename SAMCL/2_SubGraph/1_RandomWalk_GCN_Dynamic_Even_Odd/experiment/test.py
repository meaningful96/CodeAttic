import json

path = "/home/youminkk/Model_Experiment/2_SubGraph/1_RandomWalk_Dynamic/data/WN18RR/valid.txt.json"
data = json.load(open(path, 'r', encoding='utf-8'))
batch_size = 1024
print("Step per Epoch: {}".format(len(data) * 2 //batch_size))
subgraph = 16


step = 169
print(len(data))
print(len(data) //(subgraph))
print(step * batch_size // (subgraph*2))


a = ('01156112', 'derivationally related form', '04910973')

for triple in data:
    if triple["head_id"] == a[0] and triple['relation'] == a[1] and triple['tail_id'] == a[2]:
        print(triple)
        print("VVV")
        
    if triple["head_id"] == a[2] and triple['relation'] == a[1] and triple['tail_id'] == a[0]:
        print(triple)
        print("PPPP")
