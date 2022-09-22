# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 21:39
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MLP_regression_reproduce_v234.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

test_size = 0.25
# values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1', 'b18_v2', 'b18_v3', 'b19']
values = ['b18_v2', 'b18_v3']


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

    less10 = len(abs_Epsilon[abs_Epsilon < 30])
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
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
if __name__ == "__main__":
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

    # print(df_data1)
    # --------------------------------------
    # '''
    # b17
    # '''
    # list_result_less10 = []
    # for i in range(df_data1.shape[1]):
    #     data_feature = df_data1[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data1, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data1.shape[0] * (df_data1.shape[1] - 1) * test_size)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     #break  #测试 一次
    # print("==================================================================")
    # print("pridiction siteration:", len(MAE))  # 13*14 次
    # result_mae = sum(MAE) / len(MAE)
    # print("MAE", result_mae)
    # result_rmse = sum(RMSE) / len(RMSE)
    # print("RMSE", result_rmse)
    # result_less10 = sum(list_result_less10) / len(list_result_less10)
    # print("LESS10:", result_less10)
    # result_MAE_plot.append(result_mae)
    # result_RMSE_plot.append(result_rmse)
    # result_LESS10_plot.append(result_less10)
    # MAE.clear()
    # RMSE.clear()
    # result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0

    # --------------------------------------
    # '''
    # b17_2
    # '''
    # list_result_less10 = []
    # for i in range(df_data2.shape[1]):
    #     data_feature = df_data2[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data2, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data2.shape[0] * (df_data2.shape[1] - 1) * test_size)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     #break  #测试 一次
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
    #
    # result_mae, result_rmse, result_less10,LESS10 = 0,0,0,0

    # --------------------------------------
    # '''
    # b17_3
    # '''
    # list_result_less10 = []
    # for i in range(df_data3.shape[1]):
    #     data_feature = df_data3[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data3, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data3.shape[0] * (df_data3.shape[1] - 1) * test_size)  # 乘以 test_size
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
    #
    # result_mae, result_rmse, result_less10,LESS10 = 0,0,0,0

    # --------------------------------------
    # '''
    # b18_1
    # '''
    # list_result_less10 = []
    # for i in range(df_data4.shape[1]):
    #     data_feature = df_data4[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data4, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data4.shape[0] * (df_data4.shape[1] - 1) * test_size)  # 乘以 test_size
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

    # --------------------------------------
    # '''
    # b18_2
    # '''
    list_result_less10 = []
    for i in range(df_data5.shape[1]):
        data_feature = df_data5[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data5, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data5.shape[0] * (df_data5.shape[1] - 1) * test_size)  # 乘以 test_size
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
    #'''
    #b18_3
    #'''
    list_result_less10 = []
    for i in range(df_data6.shape[1]):
        data_feature = df_data6[:, i].reshape(-1, 1)  # 第 i 列
        data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
        for j in data_target.T:  # 对 列 进行迭代
            tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
            MAE.append(tmp_mae)
            RMSE.append(tmp_rmse)
            LESS10 += len_less10
        one_LESS10 = LESS10 / (df_data6.shape[0] * (df_data6.shape[1] - 1) * test_size)  # 乘以 test_size
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
 ###
    #b19

    # list_result_less10 = []
    # for i in range(df_data7.shape[1]):
    #     data_feature = df_data7[:, i].reshape(-1, 1)  # 第 i 列
    #     data_target = np.delete(df_data6, i, axis=1)  # del 第 i 列
    #     for j in data_target.T:  # 对 列 进行迭代
    #         tmp_mae, tmp_rmse, len_less10 = myMLPRegressor(data_feature, j.reshape(-1, 1))
    #         MAE.append(tmp_mae)
    #         RMSE.append(tmp_rmse)
    #         LESS10 += len_less10
    #     one_LESS10 = LESS10 / (df_data7.shape[0] * (df_data7.shape[1] - 1) * test_size)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     break  # 测试 一次
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
    # result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0
    #


##plot
    print("---------------------------------------------")
    print("result_MAE_plot",result_MAE_plot)
    print("result_RMSE_plot", result_RMSE_plot)
    print("result_LESS10_plot", result_LESS10_plot)


    ##figure - MAE
    plt.figure(1)
    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="blue", marker='v', linewidth=1, label="MAE")
    plt.title("MAE")
    plt.xlabel('benchmark')
    plt.ylabel('MAE_value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(0) #不显示网格线
    plt.xticks(x_ax,values)
    plt.show()

    ##figure - RMSE
    plt.figure(2)
    #x_ax = range(1, len(result_RMSE_plot) + 1)
    plt.plot(x_ax, result_RMSE_plot, color="blue", marker='v', linewidth=1, label="RMSE")
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
    plt.plot(x_ax, np.array(result_LESS10_plot)*100, color="blue", marker='v', linewidth=1, label="LESS10")
    plt.title("LESS10")
    plt.xlabel('benchmark')
    plt.ylabel('LESS30(%))')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(0)  # 不显示网格线
    plt.xticks(x_ax, values)
    plt.show()

'''
result_MAE_plot [98.01578503469811, 55.049662424799976, 16.55337208986829, 140.08156517100534, 114.52690422859027]
#要开方！ 下面这个是MSE
result_RMSE_plot [18005.63240078954, 9218.374012075514, 557.7368960290847, 36385.6816289802, 23083.026416110777]
result_LESS10_plot [0.23186666666666672, 0.6169333333333333, 0.8822666666666666, 0.18422222222222223, 0.19871604938271606]

'''