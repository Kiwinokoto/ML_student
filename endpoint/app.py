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
            query_text = request.form.get('query_text', '')

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
            # Get the data from the form
            query_text = request.form.get('user_input_text', '')

            # ! security check here

            # send request


            # Convert the data to uppercase
            uppercase_text = query_text.upper()

            # Return the result
            return render_template('index.html', user_input_text=query_text, result=uppercase_text)

        else:
            return render_template('index.html', user_input_text='Please type something', result='')

    except Exception as e:
        # Handle errors
        return str(e)

if __name__ == "__main__":
    # local
    # app.run(debug=True)

    # Run the app on 0.0.0.0 (accessible externally) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)