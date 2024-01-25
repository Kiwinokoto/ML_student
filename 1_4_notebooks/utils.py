# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import requests
import json
import pickle

# trop ?
import os, sys, random
import ast
# from zipfile import ZipFile
import numpy as np
import pandas as pd
from collections import Counter

# NLP
from bs4 import BeautifulSoup
import re, string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# modeles
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.matutils import Sparse2Corpus
from gensim import similarities

#
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor

import mlflow.pyfunc


train = pd.read_csv('./../0_data/cleaned_data/train_bow_uniques.csv', sep=',')
test = pd.read_csv('./../0_data/cleaned_data/test_bow_uniques.csv', sep=',')


def turn_str_back_into_list(df):
    """Correct the type change due to .csv export"""

    df['title_nltk'] = df['title_nltk'].apply(ast.literal_eval)
    df['body_nltk'] = df['body_nltk'].apply(ast.literal_eval)
    df['title_spacy'] = df['title_spacy'].apply(ast.literal_eval)
    df['body_spacy'] = df['body_spacy'].apply(ast.literal_eval)
    df['all_tags'] = df['all_tags'].apply(ast.literal_eval)

turn_str_back_into_list(train)
turn_str_back_into_list(test)


def preprocess_text(text):
    #Cleaning
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower().strip()

    # Tokenization
    try:
        tokens = nltk.word_tokenize(text)
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(" ".join(tokens))  # Apply RegexpTokenizer to the entire list

        # Remove punctuation (make sure, RegexpTokenizer should have done it already)
        tokens = [token for token in tokens if token not in string.punctuation]

    except Exception as e:
        print(f"Error in tokenization: {e}")
        return []

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Get part of speech for each token
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = []

    for token, pos_tag in pos_tags:
        # ! Uncommenting next line may crash the cell
        # print(f"Token: {token}, POS Tag: {pos_tag}")
        if pos_tag.startswith('V'):
            # On garde
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='v'))
            # Returns the input word unchanged if it cannot be found in WordNet.
        elif pos_tag.startswith('N'):
            # On garde
            try:
                lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='n'))
            except Exception as e:
                print(f"Error lemmatizing verb {token}: {e}")
        # Sinon on supprime

    # Read forbidden words (stopwords, too frequent, too rare) from the file
    with open('./forbidden_words.txt', 'r') as file:
        forbidden = [line.strip() for line in file]

    filtered_list = [token for token in lemmatized_tokens if token not in forbidden]

    # keep uniques
    seen_tokens = set()
    unique_tokens = []

    for token in filtered_list:
        if token not in seen_tokens:
            seen_tokens.add(token)
            if len(token) > 2:
                unique_tokens.append(token)

    return unique_tokens


def token_list_into_bow(X):
    documents = X.tolist()
    # print(documents)
    gensim_dictionary = Dictionary(documents)
    corpus = [gensim_dictionary.doc2bow(doc) for doc in documents]

    # Convert Gensim corpus to dense matrix
    bow_matrix = corpus2dense(corpus, num_terms=len(gensim_dictionary)).T

    return gensim_dictionary, bow_matrix


class SpecialKnn(mlflow.pyfunc.PythonModel):
    """A special model """

    def __init__(self, k, n=5):
        """
        Constructor method. Initializes the model with the specified value `n`.

        Parameters:
        -----------
        k : int
        """
        self.k = k # nb voisins, shortcut pour l'attribut .n_neighbors
        self.n = n # nb tags predits
        self.knn = KNeighborsRegressor(n_neighbors=k)
        self.dict_X = Dictionary()
        self.dict_y = Dictionary()

    def load_context(self, context):
        # when instance is created
        # self'fit(train_df=train, feature='title_nltk', target='all_tags') ?
        pass


    def fit(self, train_df, feature, target):
        X_train = train_df[feature].values
        y_train = train_df[target].values

        self.dict_X, X_bow_matrix = token_list_into_bow(X_train)
        self.dict_y, y_bow_matrix = token_list_into_bow(y_train)

        # Create a KNN Regressor
        self.knn.fit(X_bow_matrix, y_bow_matrix)


    def predict_tokens(self, input_text, train_df=train, target='all_tags'):
        """Prediction method for the custom model."""
        # Example query
        query_tokens = preprocess_text(input_text)
        # print(query_tokens)
        query_bow = self.dict_X.doc2bow(query_tokens)
        query_vector = corpus2dense([query_bow], num_terms=len(self.dict_X)).T

        # Find nearest neighbors
        _, indices = self.knn.kneighbors(query_vector)

        # Aggregate tags from neighbors
        neighbor_tags = [tag for i in indices.flatten() for tag in train_df.iloc[i][target]]

        # Predict tags based on most common tags among neighbors
        predicted_tags = [tag for tag, _ in Counter(neighbor_tags).most_common(n=5)]
        # 5 tags/question en moyenne mais on peut suggérer +
        # ici a ameliorer

        return predicted_tags


def main():
    main_knn = SpecialKnn('30')
    with open('main_knn.pkl', 'wb') as f:
        pickle.dump([main_knn], f, -1)

if __name__=='__main__':
    main()