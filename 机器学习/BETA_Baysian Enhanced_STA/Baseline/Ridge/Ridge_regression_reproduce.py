# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 16:51
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Ridge_regression_reproduce.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    # print(df_data1)

#--------------------------------------
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data1,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = linear3(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("len(MAE)",len(MAE))  # 13*14 次
    print("result",result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
# --------------------------------------
    for i in range(df_data2.shape[1]):
        data_feature = df_data2[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data2,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = linear3(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("len(MAE)",len(MAE))  # 13*14 次
    print("result",result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
# --------------------------------------
    for i in range(df_data3.shape[1]):
        data_feature = df_data3[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data3, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = linear3(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
# --------------------------------------
    for i in range(df_data4.shape[1]):
        data_feature = df_data4[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data4, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = linear3(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
# --------------------------------------
    for i in range(df_data5.shape[1]):
        data_feature = df_data5[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data5, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = linear3(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
# --------------------------------------
    for i in range(df_data6.shape[1]):
        data_feature = df_data6[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = linear3(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("len(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0

##plot
    print("result_MAE_plot",result_MAE_plot)

    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="green", marker='o', linewidth=1, label="MAE")
    plt.title("MAE")
    plt.xlabel('benchmark')
    plt.ylabel('MAE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


#result
'''
result_MAE_plot:
[0.12386843565165045, 75.7612244794656, 50.04634842232905, 74.13077984331956, 374.1132485566584, 100.09363098148886]


'''
