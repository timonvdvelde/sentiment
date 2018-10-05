#!/usr/bin/env python3

import sys
from preprocess import *

path_data = '../data/tokenized/test/'
path_pos = path_data + 'pos/'
path_neg = path_data + 'neg/'

path_embed = '../embeddings/'

file_embed_raw = 'glove.twitter.27B.200d.txt'
file_embed_json = 'glove.twitter.27B.200d.json'
file_review_vectors = 'review_vectors_twitter_200.json'

def get_paths(argv):
    if not len(argv) == 4:
        return False
    global path_pos
    global path_neg
    global file_embed_raw
    global file_embed_json
    global file_review_vectors

    if argv[3] == "test":
        path_data = '../data/tokenized/test/'
    elif argv[3] == "train":
        path_data = '../data/tokenized/train/'
    else:
        return False
    path_pos = path_data + 'pos/'
    path_neg = path_data + 'neg/'

    if argv[2] == "t25":
        file_embed_raw = 'glove.twitter.27B.25d.txt'
        file_embed_json = 'glove.twitter.27B.25d.json'
        file_review_vectors = '{}_vectors_twitter_25.json'.format(argv[3])
    elif argv[2] == "t200":
        file_embed_raw = 'glove.twitter.27B.200d.txt'
        file_embed_json = 'glove.twitter.27B.200d.json'
        file_review_vectors = '{}_vectors_twitter_200.json'.format(argv[3])
    elif argv[2] == "m25":
        file_embed_raw = "glove_25d_vectors.txt"
        file_embed_json = "glove_25d_vectors.json"
        file_review_vectors = '{}_vectors_movies_25.json'.format(argv[3])
    elif argv[2] == "m200":
        file_embed_raw = "glove_200d_vectors.txt"
        file_embed_json = "glove_200d_vectors.json"
        file_review_vectors = '{}_vectors_movies_200.json'.format(argv[3])
    else:
        return False

    return True

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
    if get_paths(sys.argv) == False:
        print("Run as: python bow.py [0/1/2] [t25/t200/m25/m200] [train/test]")
        print("If first time the first command line argument should be 0")
    else:
        if sys.argv[1] == '0':
            preprocess(embed=True)
        elif sys.argv[1] == '1':
            preprocess(embed=False)
        elif sys.argv[1] == '2':
            reviews, targets = load_review_vectors(file_review_vectors)
        else:
            print('Specify what to do')
