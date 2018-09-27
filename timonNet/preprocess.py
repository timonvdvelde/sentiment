#!/usr/bin/env python3

import json

path_embed = '../embeddings/'
file_embed_raw = 'glove.twitter.27B.25d.txt'
file_embed_json = 'glove.twitter.27B.25d.json'


def embed_to_json(file_embed_raw, file_embed_json):
    """
    Converts raw embeddings to JSON file and stores it.
    """
    table = {}

    with open(file_embed_raw) as file:
        for line in file:
            values = line.split()
            float_values = [float(val) for val in values[1:]]
            table[values[0]] = float_values

    store_json(table, file_embed_json)


def load_embed(file_embed_json):
    """
    Loads a JSON file with embeddings and converts all the vectors to Numpy
    arrays for math purposes later on.
    """
    with open(file_embed_json) as file:
        embeddings = json.load(file)

    return embeddings


def store_json(data, filename):
    """
    Stores data as JSON.
    """
    with open(filename, 'w') as file:
        json.dump(data, file)


def preprocess():
    """
    Loads embeddings to convert the review files into average word vectors. The
    results are stored in a file.
    If embed == True, also converts .txt file word embeddings to JSON and stores
    it.
    """
    print('Converting embeddings to JSON')
    embed_to_json(path_embed + file_embed_raw, path_embed + file_embed_json)
    print('Great success')


if __name__ == '__main__':
    preprocess()

