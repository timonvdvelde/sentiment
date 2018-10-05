import os
import numpy as np
import sys
import re
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from torch.utils import data
import string

def load_data_set(path):
    """
    Reads the data and returns a list
    with the movie_id, rating and its review.
    Arg:
        path: path to the reviews.
        ./aclImdb/train!test/
    Returns:
        data: list with tuples containing movie_id, rating and review.
    """

    labels = ['pos','neg']
    data = []

    for label in labels:
        directory_string = path + label
        with os.scandir(directory_string) as it:
            for file in it:
                ide_rating = file.name.split('.')[0]
                ide, rating = ide_rating.split('_')
                with open(file, 'r') as f:
                    review = f.read()
                    data.append((ide, rating, review))
    return data

def preprocess_data(data, pretrained=True):
    """
    Preprocesses the data. Depending on if we want to train
    our own word embeddings or pretrained.
    Args:
        data: list with the identity, rating and review.
        pretrained: variable that determines if we use standard preprocessing
            for pretrained word embeddings or something fancy.
    Returns:
        cleaned_data: dataframe with identity, binary sentiment labels, and preprocessed reviews.
    """
    if pretrained:
        cleaned_data = preprocess_pretrained(data)
    else:
        cleaned_data = preprocess_own(data)
    return cleaned_data

def preprocess_pretrained(data):
    """
    Very basic preprocessing. It is aimed at preparing the reviews for pretrained word embeddings.
    Args:
        data: list with the identity, rating and review.
    Returns:
        cleaned_data: DataFrame with identity, binary sentiment label and preprocessed review.
    """
    ides, ratings, reviews = [], [], []
    cleaned_data = pd.DataFrame(columns=['identity', 'rating', 'review'])
    for ide, rating, review in data:

        # Get rid of HTML
        review = bs(review, features="html5lib").get_text()
        # Keep only text
        review = re.sub("[^a-zA-Z]", " ", review)
        # To lowercase
        review = review.lower()
        # split on whitespace
        review = review.split(' ')
        # Remove empty strings
        review = [x for x in review if x != '']
        # transform output labels to binary label
        if int(rating) >= 7:
            rating = 1.0
        else:
            rating = 0.

        ides.append(ide)
        ratings.append(rating)
        reviews.append((review))

    cleaned_data['identity'] = np.asarray(ides)
    cleaned_data['rating'] = np.asarray(ratings)
    cleaned_data['review']= np.asarray(reviews)
    return cleaned_data


def preprocess_tokenized(data):
    """
    Very basic preprocessing. It is aimed at preparing the reviews for pretrained word embeddings.
    Args:
        data: list with the identity, rating and review.
    Returns:
        cleaned_data: DataFrame with identity, binary sentiment label and preprocessed review.
    """
    ides, ratings, reviews = [], [], []
    cleaned_data = pd.DataFrame(columns=['identity', 'rating', 'review'])
    for ide, rating, review in data:
        # To lowercase
        review = review.lower()

        # split on whitespace
        review = review.split(' ')

        # Remove empty strings
        review = [x for x in review if x != '']

        # transform output labels to binary label
        if int(rating) >= 7:
            rating = 1.0
        else:
            rating = 0.

        ides.append(ide)
        ratings.append(rating)
        reviews.append((review))

    cleaned_data['identity'] = np.asarray(ides)
    cleaned_data['rating'] = np.asarray(ratings)
    cleaned_data['review']= np.asarray(reviews)
    return cleaned_data

def preprocess_own(data):
    """
     More advanced preprocessing.
     It is aimed at preparing the reviews for our self trained word embeddings.
     See:
     # https://fasttext.cc/docs/en/unsupervised-tutorial.html
     # Can use t-sne to visualize word embeddings!
     # https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
    Args:
        data: list with the identity, rating and review.
    Returns:
        cleaned_data: DataFrame with identity, binary sentiment label and preprocessed review.
    """

    ides, ratings, reviews = [], [], []
    cleaned_data = pd.DataFrame(columns=['identity', 'rating', 'review'])
    tokenizer = TweetTokenizer()
    stemmer = PorterStemmer()

    for ide, rating, review in data:
        # Get rid of HTML
        review = bs(review, features="html5lib").get_text()
        # Tokenize with nltk tokenizer
        review = tokenizer.tokenize(review)
        review = [x.lower() for x in review if x not in string.punctuation]
        if int(rating) >= 7:
            rating = 1.0
        else:
            rating = 0.

        ides.append(ide)
        ratings.append(rating)
        reviews.append((review))

    cleaned_data['identity'] = np.asarray(ides)
    cleaned_data['rating'] = np.asarray(ratings)
    cleaned_data['review']= np.asarray(reviews)

    return cleaned_data


def create_train_test_data(path='./aclImdb/', pretrained=True, store_dataframe=False, tokenized=True):
    """
    Creates train and test set from preprocessed data.
    Option to store the preprocessed data as a dataframe to save time.
    Args:
        path: path to the data files.
        pretrained: Boolean that determines what type of preprocessing is used.
        store_dataframe: Store preprocessed data as pickle file (saves some time).
    """
    if pretrained and not tokenized:
        try:
            cleaned_train_data = pd.read_pickle("./pickles/preprocessed_train_data.pkl")
            cleaned_test_data = pd.read_pickle("./pickles/preprocessed_test_data.pkl")
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Obtained train and test data from stored files')
        except FileNotFoundError:
            print('Creating preprocessed train and test data:')
            train_data = load_data_set(path+'train/')
            cleaned_train_data = preprocess_data(train_data, pretrained=pretrained)
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']

            test_data = load_data_set(path+'test/')
            cleaned_test_data = preprocess_data(test_data, pretrained=pretrained)
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Done')
            if store_dataframe == True:
                print('Storing preprocessed data:')
                cleaned_train_data.to_pickle('./pickles/preprocessed_train_data.pkl')
                cleaned_test_data.to_pickle('./pickles/preprocessed_test_data.pkl')
                print('Done')
    elif not tokenized:
        try:
            cleaned_train_data = pd.read_pickle("./pickles/advanced_preprocessed_train_data.pkl")
            cleaned_test_data = pd.read_pickle("./pickles/advanced_preprocessed_test_data.pkl")
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Obtained advanced train and test data from stored files')
        except FileNotFoundError:
            print('Creating advanced preprocessed train and test data:')
            train_data = load_data_set(path+'train/')
            cleaned_train_data = preprocess_data(train_data, pretrained=pretrained)
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']

            test_data = load_data_set(path+'test/')
            cleaned_test_data = preprocess_data(test_data, pretrained=pretrained)
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Done')
            if store_dataframe == True:
                print('Storing advanced preprocessed data:')
                cleaned_train_data.to_pickle('./pickles/advanced_preprocessed_train_data.pkl')
                cleaned_test_data.to_pickle('./pickles/advanced_preprocessed_test_data.pkl')
                print('Done')


    if tokenized:
        try:
            cleaned_train_data = pd.read_pickle("./pickles/tokenized_preprocessed_train_data.pkl")
            cleaned_test_data = pd.read_pickle("./pickles/tokenized_preprocessed_test_data.pkl")
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Obtained tokenized train and test data from stored files')
        except FileNotFoundError:
            print('Creating tokenized preprocessed train and test data:')
            train_data = load_data_set(path+'train/')
            cleaned_train_data = preprocess_tokenized(train_data)
            train_X, train_y = cleaned_train_data['review'], cleaned_train_data['rating']

            test_data = load_data_set(path+'test/')
            cleaned_test_data = preprocess_tokenized(test_data)
            test_X, test_y = cleaned_test_data['review'], cleaned_test_data['rating']
            print('Done')
            if store_dataframe == True:
                print('Storing tokenized preprocessed data:')
                cleaned_train_data.to_pickle('./pickles/tokenized_preprocessed_train_data.pkl')
                cleaned_test_data.to_pickle('./pickles/tokenized_preprocessed_test_data.pkl')
                print('Done')

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)
