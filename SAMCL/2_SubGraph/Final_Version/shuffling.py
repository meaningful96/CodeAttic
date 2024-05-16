import json
import random
import numpy as np


def shuffle_dataset(data, group_size):
    total_items = len(data)
    num_groups = total_items // group_size
    # Split the data into groups
    grouped_data = [data[i:i+group_size] for i in range(0, total_items, group_size)]

    # Shuffle the order of the groups
    random.shuffle(grouped_data)

    # Flatten the list of shuffled groups
    shuffled_data_accumulated = [item for sublist in grouped_data for item in sublist]

    return shuffled_data_accumulated
