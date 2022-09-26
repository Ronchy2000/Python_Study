# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 17:16
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MLP_transfer_15nm_2idea.py
# @Software: PyCharm




#This test based on b18_v12345

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
test_size = 0.3
values = ['Corner1','Corner2','Corner3','Corner4','Corner5']



def myMLPRegressor(x,y):
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = test_size)
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
    mae = metrics.mean_absolute_error(testY, y_pred)
    rmse = metrics.mean_squared_error(testY, y_pred)
    Epsilon = testY.reshape(-1) - y_pred
    abs_Epsilon = np.maximum(Epsilon, -Epsilon)

    less10 = len(abs_Epsilon[abs_Epsilon < 10])
    print("testY:",testY.shape,"y_pred",y_pred.shape)
    print("abs_Epsilon", abs_Epsilon.shape)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("the num of less10:", less10)  # 返回的是满足条件的个数
    return mae, rmse, less10
    # plt.plot(mlp_reg.loss_curve_)
    # plt.title("Loss Curve", fontsize=14)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.show()


MAE = []
RMSE = []
LESS10 = 0

result_MAE_plot = np.empty(shape = [1,5])
result_RMSE_plot = np.empty(shape = [1,5])
result_LESS10_plot = np.empty(shape = [1,5])

if __name__ == "__main__":

    df1 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b19_15nm_5次迭代//b19_15nm_v1_x5.csv")
    df2 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b19_15nm_5次迭代//b19_15nm_v2_x5.csv")
    df3 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b19_15nm_5次迭代//b19_15nm_v3_x5.csv")
    df4 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b19_15nm_5次迭代//b19_15nm_v4_x5.csv")
    df5 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b19_15nm_5次迭代//b19_15nm_v5_x5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df5.values[:, 1:])



###one version
    list_result_less10 = []
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:, i].reshape(-1, 1)  # 第 i 列
        data_target = df_data2[:, i].reshape(-1, 1)  # 第 i 列

        tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, data_target)
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        # print("==================================================================")
        # print("pridiction siteration:", len(MAE))  # 13*14 次

    print('------------------------------------------------------')
    print('Use one version\n')
    print("MAE:",MAE)
    print("RMSE:",RMSE)
    print("less10:",list_result_less10)

    result_MAE_plot = np.array(MAE)
    result_RMSE_plot = np.array(RMSE)
    result_LESS10_plot = np.array(list_result_less10)

###two versions
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:,i].reshape(-1,1)
        tmp2 = df_data2[:,i].reshape(-1,1)
        data_feature = np.concatenate((tmp1,tmp2), axis = 1)  # 第 i 列
        data_target = df_data3[:, i].reshape(-1, 1)  # 第 i 列

        tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, data_target)
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        # print("==================================================================")
        # print("pridiction siteration:", len(MAE))  # 13*14 次

    print('------------------------------------------------------')
    print('Use two versions\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:",list_result_less10)

    result_MAE_plot = np.concatenate((result_MAE_plot.reshape(1, -1), np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot.reshape(1, -1), np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate(
        (result_LESS10_plot.reshape(1, -1), np.array(list_result_less10).reshape(1, -1)), axis=0)


###three versions
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:,i].reshape(-1,1)
        tmp2 = df_data2[:,i].reshape(-1,1)
        tmp3 = df_data3[:,i].reshape(-1,1)
        data_feature = np.concatenate((tmp1,tmp2,tmp3), axis = 1)  # 第 i 列
        data_target = df_data4[:, i].reshape(-1, 1)  # 第 i 列

        tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, data_target)
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        # print("==================================================================")
        # print("pridiction siteration:", len(MAE))  # 13*14 次

    print('------------------------------------------------------')
    print('Use three versions\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:",list_result_less10)

    result_MAE_plot = np.concatenate((result_MAE_plot, np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot, np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate((result_LESS10_plot, np.array(list_result_less10).reshape(1, -1)), axis=0)

###four versions
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:,i].reshape(-1,1)
        tmp2 = df_data2[:,i].reshape(-1,1)
        tmp3 = df_data3[:,i].reshape(-1,1)
        tmp4 = df_data3[:,i].reshape(-1,1)
        data_feature = np.concatenate((tmp1,tmp2,tmp3,tmp4), axis = 1)  # 第 i 列
        data_target = df_data5[:, i].reshape(-1, 1)  # 第 i 列

        tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, data_target)
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        # print("==================================================================")
        # print("pridiction siteration:", len(MAE))  # 13*14 次

    print('------------------------------------------------------')
    print('Use four versions\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:",list_result_less10)
    result_MAE_plot = np.concatenate((result_MAE_plot, np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot, np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate((result_LESS10_plot, np.array(list_result_less10).reshape(1, -1)), axis=0)
    print("======================================================")
    print("result_MAE_plot=", result_MAE_plot)
    print("result_RMSE_plot=", result_RMSE_plot)
    print("result_LESS10_plot=", result_LESS10_plot)


##plot
    x = [i for i in range(1,len(MAE)+1)]
    print(result_MAE_plot.shape)
    plt.figure(1)
    plt.plot(x,result_MAE_plot[0,:], label= 'one_generation')
    plt.plot(x,result_MAE_plot[1,:], label= 'two_generation')
    plt.plot(x,result_MAE_plot[2,:], label= 'three_generation')
    plt.plot(x,result_MAE_plot[3,:], label= 'four_generation')
    plt.xticks(x,values)
    plt.title("MAE")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(x, result_RMSE_plot[0,:], label='one_generation')
    plt.plot(x, result_RMSE_plot[1,:], label='two_generation')
    plt.plot(x, result_RMSE_plot[2,:], label='three_generation')
    plt.plot(x, result_RMSE_plot[3,:], label='four_generation')
    plt.xticks(x, values)
    plt.legend()
    plt.title("RMSE")
    plt.show()

    plt.figure(3)
    plt.plot(x, result_LESS10_plot[0,:], label='one_generation')
    plt.plot(x, result_LESS10_plot[1,:], label='two_generation')
    plt.plot(x, result_LESS10_plot[2,:], label='three_generation')
    plt.plot(x, result_LESS10_plot[3,:], label='four_generation')
    plt.xticks(x, values)
    plt.legend()
    plt.title("LESS10")
    plt.show()


'''
result_MAE_plot= [[10.92800129  7.89310153  8.22776285  1.73819577  2.72616657]
 [ 0.92241148  2.24969206  1.86936171  2.2004333   4.2139948 ]
 [ 7.06127678 11.56431563  2.66905206  1.5884166   3.113846  ]
 [ 6.10207738 11.63691608  4.37101679  2.8930761  12.94240306]]
result_RMSE_plot= [[265.50839024 137.27397367 189.80851797  25.56332971  34.30466647]
 [  4.55354454  23.89566821  17.57588578  20.55164964  70.99231901]
 [124.97148762 334.83956234  27.60680164  12.07422507  71.61663905]
 [ 79.45084498 343.96999316  66.53365581  30.87108448 362.85951507]]
result_LESS10_plot= [[0.63611111 0.75648148 0.74166667 0.9787037  0.93240741]
 [0.98981481 0.96944444 0.95       0.91574074 0.88611111]
 [0.78518519 0.60925926 0.94444444 0.9712963  0.96111111]
 [0.81388889 0.63796296 0.8962963  0.94444444 0.52685185]]

'''