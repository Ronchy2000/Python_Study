# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 10:01
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random Forest Regresion.py
# @Software: PyCharm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

input_file = 'Salaries.csv'
df = pd.read_csv(input_file,header = 0)  #DataFrame 格式

original_headers = list(df.columns.values)  #feature name

# remove the non-numeric columns
# df = df._get_numeric_data()
# put the numeric column names in a python list
# numeric_headers = list(df.columns.values)

from sklearn.model_selection import train_test_split

X_all = df.drop(['Position','salary'],axis = 1)
y_all = df['salary']
num_test = 0.30
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
print(predictions)
