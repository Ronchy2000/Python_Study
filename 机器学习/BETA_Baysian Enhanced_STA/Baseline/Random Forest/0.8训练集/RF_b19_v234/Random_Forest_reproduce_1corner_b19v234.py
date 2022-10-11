# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 14:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random_Forest_reproduce_1corner_b19v234.py
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

test_size = 0.3
values = [ 'b19_v2','b19_v3','b19_v4']

def RF_regression(data_feature,data_target):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_feature.reshape(-1, 1), data_target.reshape(-1, 1),
                                                    test_size = test_size)  # 25% 作为测试集
    # model = RandomForestRegressor()
    model = RandomForestRegressor(n_estimators=10,max_depth=5,bootstrap=True)



    model.fit(Xtrain, Ytrain)
    ypred = model.predict(Xtest)
    rmse = mean_squared_error(Ytest, ypred)
    mae = mean_absolute_error(Ytest, ypred)

    Epsilon = Ytest.reshape(-1) - ypred.reshape(-1)
    abs_Epsilon = np.maximum(Epsilon, -Epsilon)

    less10 = len(abs_Epsilon[abs_Epsilon < 30])
    print("testY:", Ytest.shape, "y_pred", ypred.shape)
    print("abs_Epsilon", abs_Epsilon.shape)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("the num of less10:", less10)  # 返回的是满足条件的个数
    return mae, rmse, less10

MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
if __name__ == "__main__":
    df1 = pd.read_csv("../../../Benchmark/Benchmark/b19_VTL2x5.csv")
    df2 = pd.read_csv("../../../Benchmark/Benchmark/b19_VTL3x5.csv")
    df3 = pd.read_csv("../../../Benchmark/Benchmark/b19_VTL4x5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    # --------------------------------------
    # '''
    # b19_v2
    # '''
    list_result_less10 = []
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data1, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data1.shape[0] * (df_data1.shape[1] - 1) * test_size)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        break  # 测试 一次
    print("==================================================================")
    print("pridiction siteration:", len(MAE))  # 13*14 次
    result_mae = sum(MAE) / len(MAE)
    print("MAE", result_mae)
    result_rmse = sum(RMSE) / len(RMSE)
    print("RMSE", result_rmse)
    result_less10 = sum(list_result_less10) / len(list_result_less10)
    print("LESS10:", result_less10)
    result_MAE_plot.append(result_mae)
    result_RMSE_plot.append(result_rmse)
    result_LESS10_plot.append(result_less10)
    MAE.clear()
    RMSE.clear()
    result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0


    # --------------------------------------
    # '''
    # b19_v3
    # '''
    list_result_less10 = []
    for i in range(df_data2.shape[1]):
        data_feature = df_data2[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data2, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data2.shape[0] * (df_data2.shape[1] - 1) * test_size)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        break  #测试 一次
    print("==================================================================")
    print("pridiction siteration:", len(MAE))  # 13*14 次
    result_mae = sum(MAE) / len(MAE)
    print("MAE", result_mae)

    result_rmse = sum(RMSE) / len(RMSE)
    print("RMSE", result_rmse)

    result_less10 = sum(list_result_less10) / len(list_result_less10)
    print("LESS10:", result_less10)

    result_MAE_plot.append(result_mae)
    result_RMSE_plot.append(result_rmse)
    result_LESS10_plot.append(result_less10)
    MAE.clear()
    RMSE.clear()

    result_mae, result_rmse, result_less10,LESS10 = 0,0,0,0

    # --------------------------------------
    # '''
    # b19_v4
    # '''
    list_result_less10 = []
    for i in range(df_data3.shape[1]):
        data_feature = df_data3[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data3, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data3.shape[0] * (df_data3.shape[1] - 1) * test_size)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        break  # 测试 一次
    print("==================================================================")
    print("pridiction siteration:", len(MAE))  # 13*14 次
    result_mae = sum(MAE) / len(MAE)
    print("MAE", result_mae)

    result_rmse = sum(RMSE) / len(RMSE)
    print("RMSE", result_rmse)

    result_less10 = sum(list_result_less10) / len(list_result_less10)
    print("LESS10:", result_less10)

    result_MAE_plot.append(result_mae)
    result_RMSE_plot.append(result_rmse)
    result_LESS10_plot.append(result_less10)
    MAE.clear()
    RMSE.clear()

    result_mae, result_rmse, result_less10,LESS10 = 0,0,0,0






##plot
    print("---------------------------------------------")
    print("result_MAE_plot",result_MAE_plot)
    print("result_RMSE_plot", result_RMSE_plot)
    print("result_LESS10_plot", result_LESS10_plot)

    #values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18', 'b19']

    ##figure - MAE
    plt.figure(1)
    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="red", marker='s', linewidth=1, label="MAE")
    plt.title("MAE")
    plt.xlabel('benchmark')
    plt.ylabel('MAE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(0) #不显示网格线
    plt.xticks(x_ax, values)
    plt.show()

    ##figure - RMSE
    plt.figure(2)
    #x_ax = range(1, len(result_RMSE_plot) + 1)
    plt.plot(x_ax, result_RMSE_plot, color="red", marker='s', linewidth=1, label="RMSE")
    plt.title("RMSE")
    plt.xlabel('benchmark')
    plt.ylabel('RMSE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(0)  # 不显示网格线
    plt.xticks(x_ax, values)
    plt.show()

    ##figure - LESS10
    plt.figure(3)
    # x_ax = range(1, len(result_RMSE_plot) + 1)
    plt.plot(x_ax, np.array(result_LESS10_plot)*100, color="red", marker='s', linewidth=1, label="LESS10")
    plt.title("LESS10")
    plt.xlabel('benchmark')
    plt.ylabel('LESS10(%))')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(0)  # 不显示网格线
    plt.xticks(x_ax, values)
    plt.show()

'''
---------------------------------------------
result_MAE_plot [75.30315422630774, 65.20723834311937, 16.656610685776467, 163.75805868038617, 64.32663452319106, 53.11690632519584, 116.03983927024045]
result_RMSE_plot [11030.536624955503, 10789.335564273795, 584.1747463025556, 49389.28355687186, 7601.87534882711, 4826.049597582013, 23019.817314888878]
result_LESS10_plot [0.29, 0.31666666666666665, 0.8844444444444445, 0.19733796296296297, 0.3244212962962963, 0.3398148148148148, 0.1787037037037037]

'''

