# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 20:14
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : RF_Corner.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

feature_name,target_name =[],[]

df = pd.read_csv("mydata1.csv")
header = list(df.columns.values)
feature_name = header[:-1]
target_name.append( header[-1] )
print("feature_name:",feature_name)
print("target_name:",target_name)

X_all = df.drop(target_name,axis = 1)
y_all = df[target_name]

x = np.array(X_all)
y = np.array(y_all)
# print(X_all)
# print(y_all)
Xtrain , Xtest , Ytrain , Ytest = train_test_split(x,y,test_size = 0.3)  #30% 作为测试集

tree = RandomForestRegressor(n_estimators = 100, random_state = 42)
tree.fit(Xtrain,Ytrain)
target_predicted = tree.predict(Xtest)
# print("target_predicted:",target_predicted)
print(len(Xtest),len(Ytest))

errors = abs(Ytest-target_predicted)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.') #MAE





