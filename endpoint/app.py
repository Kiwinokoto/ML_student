# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import requests
import json


app = Flask(__name__)

@app.route('/predict/', methods=['POST'])
def endpoint():
    try:
        if request.method == 'POST':
            # Get the data from the form
            query_text = request.form['query_text']

            # ! security check here

            # Convert the data to uppercase
            uppercase_text = query_text.upper()

            # Return the result
            return uppercase_text

    except Exception as e:
        # Handle errors
        return str(e)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # Handle form submission here
        input_text = request.form.get('inputText', '')

        # Perform any processing or make predictions here
        result = f'Result for input: {input_text}'

        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == "__main__":
    # local
    # app.run(debug=True)

    # Run the app on 0.0.0.0 (accessible externally) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)