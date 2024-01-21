# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

# import pickle
import joblib


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_topics():
    try:
        # Get content from the POST request
        content = request.form['content']

        # Call your topics prediction function
        predicted_topics = custom_knn(content)

        # save the model to disk
        # pickle first, test joblib for knn
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

        # Export your model to a file
        with open('my_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # save the model to disk
        filename = 'finalized_model.sav'
        joblib.dump(model, filename)

        # some time later...

        # load the model from disk
        loaded_model = joblib.load(filename)
        result = loaded_model.score(X_test, Y_test)
        print(result)

        # some time later...

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)

        # Return the result as JSON
        return jsonify({'result': predicted_topics})

    except Exception as e:
        return jsonify({'error': str(e)})