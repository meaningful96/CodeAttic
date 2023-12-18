import json
import time
import datetime
#"""
print("Start Concatenation!!")

start_time = time.time()
path = "/home/youminkk/Model_Experiment/1_Random_Walk/data/WN18RR/random_walk_train_32.json"

with open(path, 'r') as file:
    data = json.load(file)

flattened = [item for sublist in data for item in sublist]
output_path = "/home/youminkk/Model_Experiment/1_Random_Walk/data/WN18RR/random_walk_train_32.txt.json"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(flattened, f, ensure_ascii=False, indent=4)

end_time = time.time()
sec = end_time - start_time

print("Taking Time for stacking:", datetime.timedelta(seconds = sec))
print("Stacking Done!!")
#"""
#"""

print("Start Concatenation!!")

start_time = time.time()
path = "/home/youminkk/Model_Experiment/1_Random_Walk/data/WN18RR/random_walk_valid_32.json"
with open(path, 'r') as file:
    data = json.load(file)

flattened = [item for sublist in data for item in sublist]
output_path = "/home/youminkk/Model_Experiment/1_Random_Walk/data/WN18RR/random_walk_valid_32.txt.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(flattened, f, ensure_ascii=False, indent=4)

end_time = time.time()
sec = end_time - start_time

print("Taking Time for stacking:", datetime.timedelta(seconds = sec))
print("Stacking Done!!")
#"""


