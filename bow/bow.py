#!/usr/bin/env python3

import sys
from preprocess import *

path_data = '../dataset/train/'
path_pos = path_data + 'pos/'
path_neg = path_data + 'neg/'

path_embed = '../embeddings/'
file_embed_raw = 'glove.twitter.27B.25d.txt'
file_embed_json = 'glove.twitter.27B.25d.json'

file_review_vectors = 'review_vectors.json'


def preprocess(embed=False):
    """
    Loads embeddings to convert the review files into average word vectors. The
    results are stored in a file.
    If embed == True, also converts .txt file word embeddings to JSON and stores
    it.
    """
    if embed:
        print('Converting to JSON')
        embed_to_json(path_embed + file_embed_raw, path_embed + file_embed_json)
        print('Great success')

    print('Loading JSON dictionary')
    embeddings = load_embed(path_embed + file_embed_json)
    print('Done')

    review_vectors = []
    print('Computing positive review vectors')
    all_reviews_to_vectors(embeddings, path_pos, 1, review_vectors)
    print('Computing negative review vectors')
    all_reviews_to_vectors(embeddings, path_neg, 0, review_vectors)
    print('Done')

    store_json(review_vectors, file_review_vectors)


if __name__ == '__main__':
    if sys.argv[1] == '0':
        preprocess(embed=True)
    elif sys.argv[1] == '1':
        preprocess(embed=False)
    elif sys.argv[1] == '2':
        reviews, targets = load_review_vectors(file_review_vectors)
    else:
        print('Specify what to do')

