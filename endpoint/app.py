# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import requests
import json


app = Flask(__name__)


@app.route('/predict/', methods=['GET', 'POST'])
def endpoint():
    try:
        if request.method == 'POST':
            # Get the data from the form
            query_text = request.form.get('query_text')

            # ! security check here

            # Convert the data to uppercase
            uppercase_text = query_text.upper()

            # Return the result
            return uppercase_text, 200

        else:
            return 'hello world!'

    except Exception as e:
        # Handle errors
        return str(e)


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            if request.form.get('user_input_text'): # on a un input, il faut appeler le modele
                # Get the data from the form
                user_input_text = request.form.get('user_input_text')

                # ! security check here

                # Make a synchronous request to /predict/ endpoint
                answer = requests.post('https://www.kiwinokoto.com/predict/', data={'query_text': user_input_text})
                result = answer.text

                # Return the result
                return render_template('index.html', user_input_text=user_input_text, result=result)

            else: # bizarre
                return 'hello world ???'

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