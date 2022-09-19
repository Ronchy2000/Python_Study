# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 14:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random_Forest_reproduce_allCombination.py
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
                                                    test_size=0.25)  # 25% 作为测试集
    model = RandomForestRegressor()
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
    # df1 = pd.read_csv("..\\..\\Benchmark\\timing1500x14.csv")
    # df2 = pd.read_csv("..\\..\\Benchmark\\timing3700x14.csv")
    # df3 = pd.read_csv("..\\..\\Benchmark\\timing9500x14.csv")
    # df4 = pd.read_csv("..\\..\\Benchmark\\timing20000x14.csv")
    # df5 = pd.read_csv("..\\..\\Benchmark\\timing50000x14.csv")
    # df6 = pd.read_csv("..\\..\\Benchmark\\timing100000x14.csv")
    df1 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b17_VTL1x5.csv")
    df2 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b17_VTL2x5.csv")
    df3 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b17_VTL3x5.csv")

    df4 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b18_VTLx5.csv")
    df5 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b19_VTLx5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df5.values[:, 1:])
    # df_data3 = np.array(df3.values[:, 1:])
    # df_data4 = np.array(df4.values[:, 1:])
    # df_data5 = np.array(df5.values[:, 1:])
    # df_data6 = np.array(df6.values[:, 1:])

    # --------------------------------------
    # '''
    # b17
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
        one_LESS10 = LESS10 / (df_data1.shape[0] * (df_data1.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        #break  # 测试 一次
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
    # b18
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
        one_LESS10 = LESS10 / (df_data2.shape[0] * (df_data2.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        #break  #测试 一次
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
    # b19
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
        one_LESS10 = LESS10 / (df_data3.shape[0] * (df_data3.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
       #break  # 测试 一次
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
    # b20
    # '''
    list_result_less10 = []
    for i in range(df_data4.shape[1]):
        data_feature = df_data4[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data4, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data4.shape[0] * (df_data4.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        #break  # 测试 一次
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
    # b21
    # '''
    list_result_less10 = []
    for i in range(df_data5.shape[1]):
        data_feature = df_data5[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data5, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data5.shape[0] * (df_data5.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        #break  # 测试 一次
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

    # # --------------------------------------
    # # '''
    # # b22
    # # '''
    # list_result_less10 = []
    # for i in range(df_data6.shape[1]):
    #     data_feature = df_data6[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data6.shape[0] * (df_data6.shape[1] - 1) * 0.25)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     #break  # 测试 一次
    # print("==================================================================")
    # print("pridiction siteration:", len(MAE))  # 13*14 次
    # result_mae = sum(MAE) / len(MAE)
    # print("MAE", result_mae)
    #
    # result_rmse = sum(RMSE) / len(RMSE)
    # print("RMSE", result_rmse)
    #
    # result_less10 = sum(list_result_less10) / len(list_result_less10)
    # print("LESS10:", result_less10)
    #
    # result_MAE_plot.append(result_mae)
    # result_RMSE_plot.append(result_rmse)
    # result_LESS10_plot.append(result_less10)
    # MAE.clear()
    # RMSE.clear()
    # result_mae, result_rmse, result_less10,LESS10 = 0,0,0,0
    #







##plot
    print("---------------------------------------------")
    print("result_MAE_plot",result_MAE_plot)
    print("result_RMSE_plot", result_RMSE_plot)
    print("result_LESS10_plot", result_LESS10_plot)

    values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18', 'b19']

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
result_MAE_plot [0.19508149774853237, 86.72016416203127, 57.505882926711124, 86.00809835016133, 432.3321602893996, 112.21760593331265]
result_RMSE_plot [0.07939115520776713, 11014.576478011777, 4812.767409537852, 10831.0829966438, 273429.7573917104, 18226.563149038764]
result_LESS10_plot [1.0, 0.06710614935608303, 0.09690236717263745, 0.06684809209624684, 0.013343139638295056, 0.04964439080868164]
'''


