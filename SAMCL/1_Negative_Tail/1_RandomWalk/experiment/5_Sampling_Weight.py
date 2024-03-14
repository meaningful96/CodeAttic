import json

def read_json(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        data =json.load(file)

    return data


def Sampling_Weight()
