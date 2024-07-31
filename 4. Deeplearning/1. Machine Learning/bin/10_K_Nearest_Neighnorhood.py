"""
Created on meaningful96

DL Project
"""

## K-NN Algorithm

## Step 1. Dataset

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

#%%
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

plt.close("all")

## Step 2. Plot all data
data = np.array(dataset)
x = data[:,0]
y = data[:,1]
plt.plot(x,y,'r.')

#%%
## Step 3. Plot group by group
group_0 = data[:5]
group_1 = data[5:]

plt.plot(group_0[:,0], group_0[:,1],'ro')
plt.plot(group_1[:,0], group_1[:,1],'bo')

#%%

# calculate the Euclidean distance between two vectors
# row = [x, y, type]
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	for neighbor in neighbors:
		print(neighbor)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction


row0 = [3,3,0]

prediction = predict_classification(dataset, row0, 3)
print('Expected %d, Got %d.' % (row0[-1], prediction))


row1 = [6,5,0]
prediction = predict_classification(dataset, row1, 3)
print('Expected %d, Got %d.' % (row1[-1], prediction))    
