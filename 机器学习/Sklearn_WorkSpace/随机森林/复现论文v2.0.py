# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 19:47
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 复现论文v2.0.py
# @Software: PyCharm



import numpy as np
from numpy import mean
from numpy import std

import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_regression  #生成回归数据，没有用到
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
###
#数据导入及处理

df = pd.read_csv("mydata2_corner1-corner14.csv")

header = list(df.columns.values)
# feature_name = header[:]
# print("feature_name:\n\n",feature_name)

feature_name = header[0] #Corner1 作为 Seed
#belta,gama
belta = df[feature_name]
gama = df.drop(feature_name,axis = 1)

x = np.array(belta)
y = np.array(gama)

# x = preprocessing.scale(x)  #标准化
# y = preprocessing.scale(y)

# x = x.reshape(-1,1)  #1D  -> 2D

# print(x)
# print("---------------------------------")
# print(y)

Xtrain , Xtest , Ytrain , Ytest = train_test_split(x.reshape(-1,1),y[:,0].reshape(-1,1),test_size = 0.3)  #30% 作为测试集
'''
print(Xtrain)
print("---------------------------------")
print(Xtest)
print("---------------------------------")
print(Ytrain)
print("---------------------------------")
print(Ytest)
'''

MAE,MSE,RMSE = [], [], []
for i in range(y.shape[1]):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x.reshape(-1, 1), y[:, i].reshape(-1, 1),
                                                    test_size=0.3)  # 30% 作为测试集
    ###建模
    model = RandomForestRegressor()
    model.fit(Xtrain,Ytrain)
    score = model.score(Xtrain, Ytrain)
    print("R-squared:", score)

    ypred = model.predict(Xtest)
    mse = mean_squared_error(Ytest, ypred)
    mae = mean_absolute_error(Ytest, ypred)

    print("MSE: ", mse)
    print("RMSE: ", mse*(1/2.0))  #均方根误差
    print("MAE:",mae)
    MAE.append(mae)
    MSE.append(mse)
    RMSE.append(mse*(1/2.0))
    print("-------------------")

    ###Plot
    #画图 -> predict 与 本身相比
    # x_ax = range(len(Ytest))
    # plt.plot(x_ax, Ytest, linewidth=1, label="original")
    # plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    # plt.title("y-test and y-predicted data")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(loc='best',fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show()

print("=============================================================")
print("MAE:",MAE)
print("RMSE:",RMSE)
print("MSE:",MSE)

#查找最大数
max_MAE = MAE[0]
for i in MAE:
    if i > max_MAE:
        max_MAE = i
print("max_MAE:",max_MAE)
print("max_MAE.index:",MAE.index(max_MAE))

max_RMSE = RMSE[0]
for i in RMSE:
    if i > max_RMSE:
        max_RMSE = i
print("max_RMSE:",max_RMSE)
print("max_RMSE.index:",RMSE.index(max_RMSE))

max_MSE = MSE[0]
for i in MSE:
    if i > max_MSE:
        max_MSE = i
print("max_MSE:",max_MSE)
print("max_MSE.index:",MSE.index(max_MSE))

