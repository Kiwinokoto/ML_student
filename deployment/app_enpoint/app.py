# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

import pickle
# import joblib
import os, sys, random
import ast
# from zipfile import ZipFile
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from pprint import pprint

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

# Feature engineering
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import similarities
from gensim.models.ldamulticore import LdaMulticore

from sklearn.model_selection import GridSearchCV

# Preprocessing, identique à celui des autres docs

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
    with open('./../data/cleaned_data/forbidden_words.txt', 'r') as file:
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


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def hello():
    return "Hello World!"

def predict_topics():
    try:
        # Get content from the POST request
        content = request.form['content']

        tokens = preprocess_text(content)

        # load the model from disk using pickle
        custom_knn = pickle.load(open('my_knn.pkl', 'rb'))

        predicted_topics = custom_knn(tokens)


        # some time later...

        # load the model from disk
        # loaded_model = joblib.load(filename)
        # result = loaded_model.score(X_test, Y_test)

        # some time later...

        # Return the result as JSON
        return jsonify({'result': predicted_topics})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == "__main__":
    # local
    # app.run(debug=True)

    # Run the app on 0.0.0.0 (accessible externally) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)