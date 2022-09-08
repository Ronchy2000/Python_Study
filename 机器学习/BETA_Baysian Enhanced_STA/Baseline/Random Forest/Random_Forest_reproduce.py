# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 14:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random_Forest_reproduce.py
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

def RF_regression(data_feature,data_target):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_feature.reshape(-1, 1), data_target.reshape(-1, 1),
                                                    test_size=0.3)  # 30% 作为测试集
    model = RandomForestRegressor()
    model.fit(Xtrain, Ytrain)
    ypred = model.predict(Xtest)
    # mse = mean_squared_error(Ytest, ypred)
    MAE = mean_absolute_error(Ytest, ypred)
    print("MAE", MAE)
    return MAE

MAE = []
result_MAE_plot = []
if __name__ == "__main__":
    df1 = pd.read_csv("..\\..\\Benchmark\\timing1500x14.csv")
    df2 = pd.read_csv("..\\..\\Benchmark\\timing3700x14.csv")
    df3 = pd.read_csv("..\\..\\Benchmark\\timing9500x14.csv")
    df4 = pd.read_csv("..\\..\\Benchmark\\timing20000x14.csv")
    df5 = pd.read_csv("..\\..\\Benchmark\\timing50000x14.csv")
    df6 = pd.read_csv("..\\..\\Benchmark\\timing100000x14.csv")
    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df5.values[:, 1:])
    df_data6 = np.array(df6.values[:, 1:])

    # --------------------------------------
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data1,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = RF_regression(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"len(MAE)",len(MAE))  # 13*14 次
    print("result",result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data2.shape[1]):
        data_feature = df_data2[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data2, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data3.shape[1]):
        data_feature = df_data3[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data3, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data4.shape[1]):
        data_feature = df_data4[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data4, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data5.shape[1]):
        data_feature = df_data5[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data5, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data6.shape[1]):
        data_feature = df_data6[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0


##plot
    print("result_MAE_plot",result_MAE_plot)

    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="red", marker='s', linewidth=1, label="MAE")
    plt.title("MAE")
    plt.xlabel('benchmark')
    plt.ylabel('MAE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


'''
result_MAE_plot [0.20476553814296486, 86.91918068314574, 57.610878828294396, 85.33625341196115, 432.6371416123396, 112.52693003915876]


'''


