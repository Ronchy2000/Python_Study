# -*- coding: utf-8 -*-
# @Time    : 2022/8/26 15:44
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Ridge_regression models.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def linear3(data_feature,data_target):
    x_train,x_test,y_train,y_test = train_test_split(data_feature,data_target,random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge()
    estimator.fit(x_train,y_train)

    print("Ridge Regression 权重系数:",estimator.coef_)
    print("Ridge Regression 偏置:",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    #print("预测值:",y_predict)
    RMSE =  mean_squared_error(y_test,y_predict)
    MAE = mean_absolute_error(y_test,y_predict)
    print("RMSE",RMSE)
    print("MAE", MAE)
    return MAE


MAE = []
if __name__ == "__main__":
    df = pd.read_csv("timing1500x14.csv")
    df_data = np.array( df.values[:,1:] )
#单次预测
    # data_feature = df_data[:,[1,3,4,5,6,7,8]]
    # data_target = df_data[:,2].reshape(-1,1)
    # linear3(data_feature,data_target)
#set dominant corners to 1, investigate all the combinations
# ,obtained the average value of these metrices
    for i in range(df_data.shape[1]):
        data_feature = df_data[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = linear3(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"len(MAE)",len(MAE))  # 13*14 次
    print("result",result)

##  result 0.12386843565163529