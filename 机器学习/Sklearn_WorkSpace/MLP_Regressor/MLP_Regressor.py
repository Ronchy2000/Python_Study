# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MLP_Regressor.py
# @Software: PyCharm
# A good tutorial: https://michael-fuchs-python.netlify.app/2021/02/10/nn-multi-layer-perceptron-regressor-mlpregressor/

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('timing1500x14.csv')
x =
y =

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
sc=StandardScaler()
scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_reg = MLPRegressor(hidden_layer_sizes=(150,100,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(trainX_scaled, trainY)

y_pred = mlp_reg.predict(testX_scaled)