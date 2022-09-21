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

test_size = 0.3
values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1', 'b18_v2', 'b18_v3', 'b19']

def RF_regression(data_feature,data_target):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_feature, data_target.reshape(-1, 1),
                                                    test_size = test_size)  # 25% 作为测试集
    # model = RandomForestRegressor()
    model = RandomForestRegressor(n_estimators=10,max_depth=2,bootstrap=True)



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
first_corner, second_corner = 1, 2
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

    df4 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b18_VTL1x5.csv")
    df5 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b18_VTL2x5.csv")
    df6 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b18_VTL3x5.csv")

    df7 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b19_VTLx5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df6.values[:, 1:])
    df_data6 = np.array(df5.values[:, 1:])
    df_data7 = np.array(df7.values[:, 1:])


    # --------------------------------------
    # '''
    # b17_v1
    # '''
    list_result_less10 = []
    data_feature = df_data1[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data1, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)
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
    # b17_v2
    # '''
    list_result_less10 = []
    data_feature = df_data2[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data2, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)

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
    # b17_v3
    # '''
    list_result_less10 = []
    data_feature = df_data3[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data3, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)
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
    # b18_v1
    # '''
    list_result_less10 = []
    data_feature = df_data4[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data4, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)
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
    # b18_v2
    # '''
    list_result_less10 = []
    data_feature = df_data5[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data5, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)
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
    # '''
    # b18_v3
    # '''
    list_result_less10 = []
    data_feature = df_data6[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data6, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)

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
    #

    # '''
    # b19
    # '''
    list_result_less10 = []
    data_feature = df_data7[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data7, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, j.reshape(-1, 1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)

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

two_Corner
result_MAE_plot [68.06072351195603, 53.24573601554987, 17.276974260715786, 177.9374211581506, 48.205539393716286, 39.265615952709666, 94.46814641919934]
result_RMSE_plot [7724.02395391465, 8916.365372767777, 549.1999669602999, 51476.92286025771, 4623.919956601782, 3151.782387341271, 15563.07618262879]
result_LESS10_plot [0.2659259259259259,  0.6666666666666666, 0.8777777777777778, 0.13657407407407407, 0.44320987654320987, 0.5439814814814815, 0.2074074074074074]

three_corner
result_MAE_plot [62.06283854431655, 63.3922500199854, 17.864698580562056, 185.24373954998092, 46.94684304695288, 39.80299308679386, 84.3032600948388]
result_RMSE_plot [6706.926502261549, 11677.464440424461, 650.669944411469, 54938.74195641561, 4366.284589219491, 3049.5216763662165, 12822.524269745367]
result_LESS10_plot [0.29555555555555557, 0.5333333333333333, 0.8555555555555555, 0.12361111111111112, 0.4525462962962963, 0.5263888888888889, 0.23189300411522634]

'''

