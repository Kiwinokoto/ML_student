# script création de l'API (endpoint), via flask
# Met aussi en ligne une interface (formulaire), permettant de tester le modèle déployé.


# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template
import requests
import pickle
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
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from sklearn.neighbors import KNeighborsRegressor

# import utils


app = Flask(__name__)


# preprocessing (remplacer par un import ?)

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


# prediction

# Load the dictionary from disk
# Get the current working directory
current_directory = os.getcwd() # pour éviter les adresses relatives (à cause de pytest)

# Construct the absolute path to the dictionary file
dictionary_path = os.path.join(current_directory, "model", "gensim_dict.pkl")
gensim_dict = pickle.load(open(dictionary_path, 'rb'))

# Load the targets
targets_path = os.path.join(current_directory, "model", "pickled_targets.pkl")
targets = pickle.load(open(targets_path, 'rb'))


def predict_tokens(model, input_vect, targets=targets):
    """Prediction method for the custom model."""

    # Find nearest neighbors
    _, indices = model.kneighbors(input_vect)

    # Aggregate tags from neighbors
    neighbor_tags = [tag for i in indices.flatten() for tag in targets[i]]

    # Predict tags based on most common tags among neighbors
    predicted_tags = [tag for tag, _ in Counter(neighbor_tags).most_common(n=5)]
    # 5 tags/question en moyenne mais on peut suggérer +
    # ici a ameliorer

    return predicted_tags


# recup models

#other
pickled_model_uri = './model/pickled_lr.pkl' # idem
model = pickle.load(open(pickled_model_uri, 'rb'))


# recup mlb
mlb_uri = './model/pickled_mlb.pkl'
mlb = pickle.load(open(mlb_uri, 'rb'))



@app.route('/predict/', methods=['GET', 'POST'])
def endpoint():
    # ! move out of route once tested (slows answer)
    try:
        # knn
        pickled_knn_uri = './model/pickled_knn.pkl' # dossier sync integration continue
        knn =  pickle.load(open(pickled_knn_uri, 'rb'))

    except Exception as e:
        # Handle errors
        return f"An error occurred while unpickling knn: {str(e)}", 500

    try:
        if request.method == 'POST':
            # Get the data from the form
            query_text = request.form.get('query_text')

            # ! security check here
            # + gerer empty if not done before ?

            # preprocess query
            query_tokens = preprocess_text(query_text)
            query_bow = gensim_dict.doc2bow(query_tokens)
            query_vector = corpus2dense([query_bow], num_terms=len(gensim_dict)).T

            # use model to predict tags

            # knn
            prediction_knn = predict_tokens(model=knn, input_vect=query_vector)

            # other models
            topics = model.predict(query_vector)
            # Inverse transform predicted labels
            predicted_tags_tuple = mlb.inverse_transform(topics)
            prediction_lr = list(predicted_tags_tuple[0]) # mlb retourne un array

            # Return the result
            return [str(prediction_knn), str(prediction_lr)], 200
            # Flask automatically converts Python dictionaries or lists into JSON responses.

        else:
            return 'hello world :)'

    except Exception as e:
        # Handle errors
        return f"An error occurred while processing request: {str(e)}", 500


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            if request.form.get('user_input_text'): # on a un input, il faut appeler le modele
                # Get the data from the form
                user_input_text = request.form.get('user_input_text')

                # ! security check here

                # Make a synchronous request to /predict/ endpoint
                try:
                    answer = requests.post('https://www.kiwinokoto.com/predict/', data={'query_text': user_input_text})
                    result_knn = answer.json()[0]
                    result_lr = answer.json()[1]
                    result = [result_knn, result_lr]
                except Exception as e:
                    return f"An error occurred while during request: {str(e)}", 500

                # Return the result
                return render_template('index.html', user_input_text=user_input_text, result=result)

            else: # sent empty form
                return render_template('index.html', user_input_text='PLEASE TYPE SOMETHING !!', result='')

        else: # first time
            return render_template('index.html', user_input_text='Please type something', result='')

    except Exception as e:
        # Handle errors
        return str(e)


if __name__ == "__main__":
    # local
    # app.run(debug=True)

    # Run the app on 0.0.0.0 (accessible externally) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)