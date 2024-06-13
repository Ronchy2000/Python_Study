# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 10:48
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 随机森林_Corner跑起来了.py
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
# print(X_all)
# print(y_all)
Xtrain , Xtest , Ytrain , Ytest = train_test_split(X_all,y_all,test_size = 0.3)  #30% 作为测试集

tree = DecisionTreeRegressor()
tree.fit(Xtrain,Ytrain)
target_predicted = tree.predict(Xtest)
print("target_predicted:",target_predicted)
