# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 21:39
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MLP_regression_reproduce.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics


def myMLPRegressor(x,y):
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
    sc=StandardScaler()
    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    testX_scaled = scaler.transform(testX)
    #MLP_Regressor
    mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                           max_iter=300, activation='relu',
                           solver='adam')

    mlp_reg.fit(trainX_scaled, trainY)
    y_pred = mlp_reg.predict(testX_scaled)

    #print('RMSE:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))
    MAE = metrics.mean_absolute_error(testY, y_pred)
    print('MAE :',MAE)
    return MAE
    # plt.plot(mlp_reg.loss_curve_)
    # plt.title("Loss Curve", fontsize=14)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.show()

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

    # --------------------------------------
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data1,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = myMLPRegressor(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"\nlen(MAE)",len(MAE))  # 13*14 次
    print("result",result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0

    # --------------------------------------
    for i in range(df_data2.shape[1]):
        data_feature = df_data2[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data2, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "\nlen(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0

    # --------------------------------------
    for i in range(df_data3.shape[1]):
        data_feature = df_data3[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data3, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "\nlen(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0

# --------------------------------------
    for i in range(df_data4.shape[1]):
        data_feature = df_data4[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data4,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = myMLPRegressor(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"\nlen(MAE)",len(MAE))  # 13*14 次
    print("result",result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data5.shape[1]):
        data_feature = df_data5[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data5, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "\nlen(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0
    # --------------------------------------
    for i in range(df_data6.shape[1]):
        data_feature = df_data6[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp)
    result = sum(MAE) / len(MAE)
    print("MAE", MAE, "\nlen(MAE)", len(MAE))  # 13*14 次
    print("result", result)

    result_MAE_plot.append(result)
    MAE.clear()
    result = 0


    ##plot
    print("result_MAE_plot", result_MAE_plot)

    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="blue", marker='v', linewidth=1, label="MAE")
    plt.title("MAE")
    plt.xlabel('benchmark')
    plt.ylabel('MAE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()