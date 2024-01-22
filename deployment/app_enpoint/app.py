# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route('/predict/') # , methods=['POST'])
def hello():
    return "Hello World!"

@app.route('/') # , methods=['POST'])
def formulaire():
    return render_template('index.php')

if __name__ == "__main__":
    # local
    # app.run(debug=True)

    # Run the app on 0.0.0.0 (accessible externally) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)