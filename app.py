from flask import Flask, request
import pandas as pd 
import numpy as np 
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('./model/model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome !'

@app.route('/predict')
def predict_note_authentication():
    """
    Let' Authenticate Bank Note
    ---
    parameters:
        -   name: variance
            in: query
            type: number
            required: true
        -   name: skewness
            in: query
            type: number
            required: true
        -   name: kurtosis
            in: query
            type: number
            required: true
        -   name: entropy
            in: query
            type: number
            required: true
    responses:
        200:
            description: The output values
    """
    variance = int(request.args.get('variance'))
    skewness = int(request.args.get('skewness'))
    kurtosis = float(request.args.get('kurtosis'))
    entropy = int(request.args.get('entropy'))

    print(type(skewness), '    Value: ', skewness)
    print('\n\n HIIII!!!!!!!\n\n')

    prediction = classifier.predict([[variance, skewness, kurtosis, entropy]])
    print('\n\n {prediction}\n\n')
    return "The predicted Value is" + str(prediction)


if __name__=='__main__':
    app.run()