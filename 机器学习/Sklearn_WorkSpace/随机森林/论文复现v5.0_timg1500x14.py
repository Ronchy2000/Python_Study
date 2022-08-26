# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 16:13
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 论文复现v4.0.py
# @Software: PyCharm


import numpy as np
from numpy import mean
from numpy import std

import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_regression  # 生成回归数据，没有用到
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

###
# 数据导入及处理

df = pd.read_csv("timing1500x14_delete_first_col.csv")
# df = pd.read_csv("mydata2_corner1-corner14.csv")

feature_name,target_name = [],[]
header = list(df.columns.values)
target_name = header[:] #拷贝给target_name

# Corner2 作为 Seed
seed_index = 0
feature_name.append(header[seed_index])
# target_name.remove(feature_name) #用index的方式删除
del target_name[seed_index]
# print("header:",header)
# print("feature_name:",feature_name)
# print("target_name:",target_name)

### belta,gama——划分featrue与target
# feature
belta = df[feature_name]
# target
gama = df.drop(feature_name, axis=1)
# print("belta",belta)
# print("gama",gama)


x = np.array(belta)
y = np.array(gama)
if x.ndim == 1:  #如果是一维，就变成二维
    x = x.reshape(-1,1)  #1D  -> 2D:针对feature只有一维的时候

# x = np.concatenate((x,y[:,11].reshape(-1,1)),axis=1)
# y = np.delete(y,11,axis=1)

# x = preprocessing.scale(x)  #标准化
# y = preprocessing.scale(y)

# x = x.reshape(-1,1)  #1D  -> 2D
# print("x.shape:",x.shape)
# print("x.ndim:",x.ndim)
# print(x)
# print("---------------------------------")
# print("y.shape:",y.shape)
# print(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y[:, 0].reshape(-1, 1), test_size=0.3)  # 30% 作为测试集
'''
print(Xtrain)
print("---------------------------------")
print(Xtest)
print("---------------------------------")
print(Ytrain)
print("---------------------------------")
print(Ytest)
'''

MAE, MSE, RMSE = [], [], []
max_mae, max_mse, max_rmse = [], [], []
min_mae, min_mse, min_rmse = [], [], []
max_iteration = 7
iteration = 0
for j in range(max_iteration):
    #iterative process
    print("feature_name:", feature_name)
    print("target_name:", target_name)
    iteration += 1
    print("iteration:",iteration)
    belta = df[feature_name]
    gama = df.drop(feature_name, axis=1)
    x = np.array(belta)
    y = np.array(gama)
    if x.ndim == 1:  # 如果是一维，就变成二维
        x = x.reshape(-1, 1)  # 1D  -> 2D:针对feature只有一维的时候
    #predicting process
    for i in range(y.shape[1]):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y[:, i].reshape(-1, 1),
                                                        test_size=0.3)  # 30% 作为测试集
        ###建模
        model = RandomForestRegressor()
        model.fit(Xtrain, Ytrain)
        score = model.score(Xtrain, Ytrain)
        # print("R-squared:", score)
        ypred = model.predict(Xtest)
        mse = mean_squared_error(Ytest, ypred)
        mae = mean_absolute_error(Ytest, ypred)
        # print("MSE: ", mse)
        # print("RMSE: ", mse * (1 / 2.0))  # 均方根误差
        # print("MAE:", mae)
        MAE.append(mae)
        MSE.append(mse)
        RMSE.append(mse * (1 / 2.0))
        # print("-------------------")
        ###Plot
        # 画图 -> predict 与 本身相比
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
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("MSE:", MSE)
    # 查找最大数
    max_MAE = MAE[0]
    min_MAE = MAE[0]
    for i in MAE:
        if i > max_MAE:
            max_MAE = i
    print("max_MAE:", max_MAE)
    print("max_MAE.index:", MAE.index(max_MAE))
#---
    for i in MAE:
        if i < max_MAE:
            min_MAE = i
    print("min_MAE:", min_MAE)
    print("min_MAE.index:", MAE.index(min_MAE))
#--------
    max_RMSE = RMSE[0]
    for i in RMSE:
        if i > max_RMSE:
            max_RMSE = i
    print("max_RMSE:", max_RMSE)
    print("max_RMSE.index:", RMSE.index(max_RMSE ))
    max_MSE = MSE[0]
    for i in MSE:
        if i > max_MSE:
            max_MSE = i
    print("max_MSE:", max_MSE)
    print("max_MSE.index:", MSE.index(max_MSE))

    # 记录，可以画图用
    max_mae.append(max_MAE)
    min_mae.append(min_MAE)
    max_rmse.append(max_RMSE)
    max_mse.append(max_MSE)


    feature_name.append(target_name[MAE.index(max_MAE)] )
    del target_name[ MAE.index(max_MAE)]
    #结束，清空list
    MAE.clear()
    MSE.clear()
    RMSE.clear()
    print('===================================')
    print("迭代完成")

x_ax = range(1,len(max_mae)+1)
plt.plot(x_ax, max_mae, linewidth=1, label="max_mae")
# plt.plot(x_ax, min_mae, linewidth=1, label="min_mae")
plt.title("MAE")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()






