# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 10:48
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 随机森林_Corner.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

df = pd.read_csv("mydata.csv")
origin_feature = df.iloc[:,1:-1]
origin_labels = df.iloc[:,-1]
# print("origin_feature:\n",origin_feature)
# print("origin_labels:\n",origin_labels)

form = pd.concat([pd.DataFrame(origin_feature), pd.DataFrame(origin_labels)],axis=1)
# print(form)

Xtrain , Xtest , Ytrain , Ytest = train_test_split(origin_feature,origin_labels,test_size = 0.3)  #30% 作为测试集

print(Xtrain.shape)
# print(Xtrain)
print('---------------------------------')
print(Ytrain.shape)
# print(Ytrain)
print(origin_feature.shape)

rfc = RandomForestClassifier()
rfc = rfc.fit(Xtrain.astype(int).astype(float),Ytrain.astype(int).astype(float))
score = rfc.score(Xtest.astype(int).astype(float),Ytest.astype(int).astype(float))

print(score)





