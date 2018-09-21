#!/usr/bin/env python3

import json
import numpy as np
import os
import sys

# For all reviews:
#   Load review
#   review_vector = []
#   for word in review:
#     find vector for word
#     review_vector += vector / nr_words
#   store review_vector

path_data = '../dataset/train/'
path_pos = path_data + 'pos/'
path_neg = path_data + 'neg/'

path_embed = '../embeddings/'
file_embed_raw = 'glove.twitter.27B.25d.txt'
file_embed_json = 'glove.twitter.27B.25d.json'

file_review_vectors = 'review_vectors.json'

embeddings = None


def embed_to_json():
    """
    Converts raw embeddings to JSON file and stores it.
    """
    table = {}

    with open(path_embed + file_embed_raw) as file:
        for line in file:
            values = line.split()
            float_values = [float(val) for val in values[1:]]
            table[values[0]] = float_values

    store_json(table, path_embed + file_embed_json)


def load_embed():
    """
    Loads a JSON file with embeddings and converts all the vectors to Numpy
    arrays for math purposes later on.
    """
    global embeddings

    with open(path_embed + file_embed_json) as file:
        embeddings = json.load(file)

        for key in embeddings.keys():
            embeddings[key] = np.array(embeddings[key])


def review_to_vector(path):
    """
    Opens a review file at some path and returns an average of all the word
    embeddings found in the document.
    """
    vector = np.zeros(0)
    i = 0

    with open(path) as file:
        for line in file:
            tokens = line.split(' ')

            for token in tokens:
                if token in embeddings:
                    if vector.size == 0:
                        vector = np.array(embeddings[token])
                    else:
                        vector += embeddings[token]
                    i += 1

    vector /= i
    return vector.tolist()


def all_reviews_to_vectors(path, label, data):
    """
    Converts all reviews under a given path to vectors and appends them labeled
    to existing data.
    """
    files = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(dirpath + '/' + filename)
        break

    for file in files:
        vector = review_to_vector(file)
        data.append([vector, label])


def store_json(data, filename):
    """
    Stores data as JSON.
    """
    with open(filename, 'w') as file:
        json.dump(data, file)


def preprocess(embed=False):
    if embed:
        print('Converting to JSON')
        embed_to_json()
        print('Great success')

    print('Loading JSON dictionary')
    load_embed()
    print('Done')

    review_vectors = []
    print('Computing positive review vectors')
    all_reviews_to_vectors(path_pos, 1, review_vectors)
    print('Computing negative review vectors')
    all_reviews_to_vectors(path_neg, 0, review_vectors)
    print('Done')

    store_json(review_vectors, file_review_vectors)    


if __name__ == '__main__':
    if sys.argv[1] == '0':
        preprocess(embed=True)
    elif sys.argv[1] == '1':
        preprocess(embed=False)
    else:
        print('Specify what to do')

