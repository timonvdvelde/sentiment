#!/usr/bin/env python3

import sys
import os
import json
from stanfordcorenlp import StanfordCoreNLP

path_embed = '../embeddings/'
#file_embed_raw = 'glove.twitter.27B.25d.txt'
#file_embed_json = 'glove.twitter.27B.25d.json'

file_embed_raw = 'glove_vectors_unsup_movies_25d_lowercase_preservelines.txt'
file_embed_json = 'glove_vectors_unsup_movies_25d_lowercase_preservelines.json'

in_paths = ['../data/train/pos/',
            '../data/train/neg/',
            '../data/test/pos/',
            '../data/test/neg/']

out_paths = ['../data/tokenized/train/pos/',
             '../data/tokenized/train/neg/',
             '../data/tokenized/test/pos/',
             '../data/tokenized/test/neg/']

stanfordpath = '../../stanford-corenlp-full-2018-02-27'


def embed_to_json(file_embed_raw, file_embed_json):
    """
    Converts raw embeddings to JSON file and stores it.
    """
    table = {}

    with open(file_embed_raw, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            try:
                float_values = [float(val) for val in values[1:]]
            except:
                continue
            table[values[0]] = float_values

    store_json(table, file_embed_json)


def load_embed(file_embed_json):
    """
    Loads a JSON file with embeddings and converts all the vectors to Numpy
    arrays for math purposes later on.
    """
    with open(file_embed_json, encoding='utf-8') as file:
        embeddings = json.load(file)

    return embeddings


def store_json(data, filename):
    """
    Stores data as JSON.
    """
    with open(filename, 'w', encoding='utf-8') as file:
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


def tokenize():
    """
    Will read reviews at input paths and store tokenized versions in output
    paths. Will be a set of sentences, each on a new line.
    """
    nlp = StanfordCoreNLP(stanfordpath)
    props = {'annotators':'tokenize,ssplit'}
    counter = 0

    for in_path, out_path in zip(in_paths, out_paths):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for dirpath, dirnames, filenames in os.walk(in_path):
            for filename in filenames:
                with open(in_path + filename) as in_file:
                    with open(out_path + filename, 'w') as out_file:
                        for line in in_file:
                            tokenized = nlp.annotate(line, properties=props)
                            tokenized = json.loads(tokenized)

                            for sentence in tokenized['sentences']:
                                for token in sentence['tokens']:
                                    out_file.write(token['word'])
                                    out_file.write(' ')

                                out_file.write('\n')

                print('Tokenized file', counter)
                counter += 1

    nlp.close()


if __name__ == '__main__':
    if sys.argv[1] == 'embed':
        preprocess()
    elif sys.argv[1] == 'token':
        tokenize()
    else:
        print('wat do?')

