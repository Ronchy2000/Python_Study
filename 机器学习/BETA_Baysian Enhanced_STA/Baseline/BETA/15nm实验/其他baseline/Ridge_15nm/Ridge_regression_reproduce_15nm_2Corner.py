# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 16:51
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Ridge_regression_reproduce_v234.py
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

test_size = 0.25
values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1', 'b18_v2', 'b18_v3', 'b19']



def linear3(data_feature,data_target):
    x_train,x_test,y_train,y_test = train_test_split(data_feature,data_target,random_state=22,test_size= test_size) #test_size 默认值：0.25

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge()
    estimator.fit(x_train,y_train)

    print("Ridge Regression 权重系数:",estimator.coef_)
    print("Ridge Regression 偏置:",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    #print("预测值:",y_predict)
    rmse =  mean_squared_error(y_test,y_predict)
    mae = mean_absolute_error(y_test,y_predict)
    Epsilon = y_test - y_predict
    abs_Epsilon = np.maximum(Epsilon,-Epsilon)
    # print("abs_Epsilon",len(abs_Epsilon))
#**********************************************
    #threshold  LESS?
    less10 = len( abs_Epsilon[abs_Epsilon < 30] )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("the num of less10:", less10)   #返回的是满足条件的个数
    return mae, rmse, less10

MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
first_corner, second_corner = 1, 2
if __name__ == "__main__":
    ###b17
    # df1 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b17_15nm_5次迭代/b17_15nm_v1_x5.csv")
    # df2 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b17_15nm_5次迭代/b17_15nm_v2_x5.csv")
    # df3 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b17_15nm_5次迭代/b17_15nm_v3_x5.csv")
    # df4 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b17_15nm_5次迭代/b17_15nm_v4_x5.csv")
    # df5 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b17_15nm_5次迭代/b17_15nm_v5_x5.csv")
    #
    # df_data1 = np.array(df1.values[:, 1:])
    # df_data2 = np.array(df2.values[:, 1:])
    # df_data3 = np.array(df3.values[:, 1:])
    # df_data4 = np.array(df4.values[:, 1:])
    # df_data5 = np.array(df5.values[:, 1:])
    ###b18
    # df1 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b18_15nm_5次迭代/b18_15nm_v1_x5.csv")
    # df2 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b18_15nm_5次迭代/b18_15nm_v2_x5.csv")
    # df3 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b18_15nm_5次迭代/b18_15nm_v3_x5.csv")
    # df4 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b18_15nm_5次迭代/b18_15nm_v4_x5.csv")
    # df5 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b18_15nm_5次迭代/b18_15nm_v5_x5.csv")
    #
    # df_data1 = np.array(df1.values[:, 1:])
    # df_data2 = np.array(df2.values[:, 1:])
    # df_data3 = np.array(df3.values[:, 1:])
    # df_data4 = np.array(df4.values[:, 1:])
    # df_data5 = np.array(df5.values[:, 1:])
    #
    ###b19
    df1 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b19_15nm_5次迭代/b19_15nm_v1_x5.csv")
    df2 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b19_15nm_5次迭代/b19_15nm_v2_x5.csv")
    df3 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b19_15nm_5次迭代/b19_15nm_v3_x5.csv")
    df4 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b19_15nm_5次迭代/b19_15nm_v4_x5.csv")
    df5 = pd.read_csv("../../../../../Benchmark/Benchmark/15nm/b19_15nm_5次迭代/b19_15nm_v5_x5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df5.values[:, 1:])

    #--------------------------------------
# '''
# b17
# '''
    list_result_less10 = []
    data_feature = df_data1[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data1, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)

    for j in data_target.T:  #对 列 进行迭代
        tmp_mae,tmp_rmse,len_less10 = linear3(data_feature,j.reshape(-1,1))
        MAE.append(tmp_mae)
        RMSE.append(tmp_rmse)
        LESS10 += len_less10
    one_LESS10 = LESS10 / (data_target.shape[0] * (data_target.shape[1]) * test_size)  # 乘以 test_size
    LESS10 = 0  #每一轮记得清零！
    list_result_less10.append(one_LESS10)
    print("==================================================================")
    print("pridiction siteration:", len(MAE))  # 13*14 次
    result_mae = sum(MAE)/len(MAE)
    print("MAE", result_mae)

    result_rmse = sum(RMSE)/len(RMSE)
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
    # b17_2
    # '''
    list_result_less10 = []
    data_feature = df_data2[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data2, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, j.reshape(-1, 1))
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
    # # --------------------------------------
    # '''
    # b17_3
    # '''
    list_result_less10 = []
    data_feature = df_data3[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data3, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, j.reshape(-1, 1))
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
    # b18_V1
    # '''
    list_result_less10 = []
    data_feature = df_data4[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data4, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, j.reshape(-1, 1))
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
    # b18_2
    # '''
    list_result_less10 = []
    data_feature = df_data5[:, [first_corner, second_corner]]  # 第 1,2列
    # print(data_feature.shape)
    data_target = np.delete(df_data5, [first_corner, second_corner], axis=1)  # del 第 1 列
    # print(data_target.shape)
    for j in data_target.T:  # 对 列 进行迭代
        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, j.reshape(-1, 1))
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


##plot
    print("---------------------------------------------")
    print("result_MAE_plot",result_MAE_plot)
    print("result_RMSE_plot", result_RMSE_plot)
    print("result_LESS10_plot", result_LESS10_plot)


    ##figure - MAE
    plt.figure(1)
    x_ax = range(1, len(result_MAE_plot) + 1)
    plt.plot(x_ax, result_MAE_plot, color="green", marker='o', linewidth=1, label="MAE")
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
    plt.plot(x_ax, result_RMSE_plot, color="green", marker='o', linewidth=1, label="RMSE")
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
    plt.plot(x_ax, np.array(result_LESS10_plot)*100, color="green", marker='o', linewidth=1, label="LESS30")
    plt.title("LESS30")
    plt.xlabel('benchmark')
    plt.ylabel('LESS30(%))')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.xticks(x_ax, values)
    plt.grid(0)  # 不显示网格线
    plt.show()

#result
'''
---------------------------------------------
Ridge Regression.
result_MAE_plot [111.30132809817754, 57.90079059731588, 17.273904424443664, 157.32547192624403, 66.36891834766786, 48.83213168636948, 125.82300093776868]
#开平方
result_RMSE_plot [20960.882955096586, 9982.765240790297, 638.8560937249046, 45542.85606003766, 9154.566199345067, 4739.935828936492, 27793.15565676238]
result_LESS10_plot [0.1806666666666667, 0.5748, 0.8429333333333332, 0.14961111111111108, 0.3958055555555556, 0.47350000000000003, 0.19407407407407407]

two_Corner
result_MAE_plot [83.43214672325949, 59.44477926043379, 17.73982119379536, 145.75583980320224, 44.942620745463636, 36.88933352082417, 90.21917856160952]
result_RMSE_plot [11121.406432391499, 10726.316636388565, 642.6313759513055, 40981.7800980818, 4405.3336638217725, 2994.6455244655285, 15327.95581387677]
result_LESS10_plot [0.19822222222222222, 0.576, 0.8346666666666667, 0.17888888888888888, 0.5109259259259259, 0.5879629629629629, 0.2773662551440329]
'''
