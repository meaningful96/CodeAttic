import json

def read_rel_label(path):
    with open(path,'r', encoding="utf-8") as json_file:
        rel_label = json.load(json_file)
        print(rel_label)
    return rel_label

rel_label = read_rel_label("data/WN18RR/relation_label.json")
print(rel_label['also see'])

