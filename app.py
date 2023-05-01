from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

@app.route('/')
def ml_test():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temperature = request.form['temperature']
    speed = request.form['speed']
    diameter = request.form['diameter']
    
    model = joblib.load('./machineLearning/model.pkl')
    features = np.array([[temperature, speed, diameter]])
    result = model.predict(features)
    return render_template('result.html', result = result)
