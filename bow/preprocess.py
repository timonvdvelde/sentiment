import numpy as np
import json
import os

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

        for key in embeddings.keys():
            embeddings[key] = np.array(embeddings[key])

    return embeddings


def review_to_vector(embeddings, path):
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
                # Converts to lowercase.
                token = token.lower()

                if token in embeddings:
                    if vector.size == 0:
                        vector = np.array(embeddings[token])
                    else:
                        vector += embeddings[token]
                    i += 1

    vector /= i
    return vector.tolist()


def all_reviews_to_vectors(embeddings, path, label, data):
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
        vector = review_to_vector(embeddings, file)
        data.append([vector, label])


def store_json(data, filename):
    """
    Stores data as JSON.
    """
    with open(filename, 'w') as file:
        json.dump(data, file)


def load_review_vectors(path):
    """
    Loads review vectors from a file. Converts the actual vectors to numpy
    arrays for calculations later.
    """
    with open(path) as file:
        review_vectors = json.load(file)

    reviews = np.zeros((len(review_vectors), len(review_vectors[0][0])))
    targets = np.zeros(len(review_vectors))

    for i in range(len(review_vectors)):
        # FIXME vector i == 4499 is emtpy.
        if len(review_vectors[i][0]) < 25:
            continue

        reviews[i] = review_vectors[i][0]
        targets[i] = review_vectors[i][1]

    return reviews, targets


