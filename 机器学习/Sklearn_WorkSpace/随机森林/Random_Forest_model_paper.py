# -*- coding: utf-8 -*-
# @Time    : 2022/8/26 20:55
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random_Forest_model_paper.py
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
if __name__ == "__main__":
    df = pd.read_csv("timing1500x14.csv")
    df_data = np.array( df.values[:,1:] )
    for i in range(df_data.shape[1]):
        data_feature = df_data[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = RF_regression(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"len(MAE)",len(MAE))  # 13*14 次
    print("result",result)




