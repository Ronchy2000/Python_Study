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

df = pd.read_csv("timing1500x14.csv")
df_data = np.array( df.values[:,1:] )
#train_data
x = df_data[:,1].reshape(-1, 1)
#test_data
y = df_data[:,2].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
model = MLPRegressor()
model.fit(X_train, y_train)

expected_y = y_test
predicted_y = model.predict(X_test)
print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))