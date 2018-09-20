#!/usr/bin/env python3

import json

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

    with open(path_embed + file_embed_json, 'w') as outfile:
        json.dump(table, outfile)


def load_embed():
    """
    Loads a JSON file with embeddings.
    """
    global embeddings

    with open(path_embed + file_embed_json) as file:
        embeddings = json.load(file)


if __name__ == '__main__':
    #print('Converting to JSON')
    #embed_to_json()
    #print('Great success')

    print('Loading JSON dictionary')
    load_embed()

    print(embeddings['legaaal'])


