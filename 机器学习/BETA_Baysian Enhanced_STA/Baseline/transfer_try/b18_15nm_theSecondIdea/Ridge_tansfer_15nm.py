# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 16:42
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Ridge_tansfer_15nm.py
# @Software: PyCharm

#This test based on b18_v12345


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

test_size = 0.3
values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1', 'b18_v2', 'b18_v3', 'b19']



def linear3(data_feature,data_target):
    x_train,x_test,y_train,y_test = train_test_split(data_feature,data_target,random_state=22,test_size= test_size) #test_size 默认值：0.25

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge()
    estimator.fit(x_train,y_train)

    # print("Ridge Regression 权重系数:",estimator.coef_)
    # print("Ridge Regression 偏置:",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    #print("预测值:",y_predict)
    rmse =  mean_squared_error(y_test,y_predict)
    mae = mean_absolute_error(y_test,y_predict)
    Epsilon = y_test - y_predict
    abs_Epsilon = np.maximum(Epsilon,-Epsilon)
    # print("abs_Epsilon",len(abs_Epsilon))
#**********************************************
    #threshold  LESS?
    less10 = len( abs_Epsilon[abs_Epsilon < 10] )

    # print("MAE:", mae)
    # print("RMSE:", rmse)
    # print("the num of less10:", less10)   #返回的是满足条件的个数
    return mae, rmse, less10



MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
if __name__ == "__main__":

    df1 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b18_15nm_5次迭代//b18_15nm_v1_x5.csv")
    df2 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b18_15nm_5次迭代//b18_15nm_v2_x5.csv")
    df3 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b18_15nm_5次迭代//b18_15nm_v3_x5.csv")
    df4 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b18_15nm_5次迭代//b18_15nm_v4_x5.csv")
    df5 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b18_15nm_5次迭代//b18_15nm_v5_x5.csv")

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

        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, data_target)
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


###two versions
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:,i].reshape(-1,1)
        tmp2 = df_data2[:,i].reshape(-1,1)
        data_feature = np.concatenate((tmp1,tmp2), axis = 1)  # 第 i 列
        data_target = df_data3[:, i].reshape(-1, 1)  # 第 i 列

        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, data_target)
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

        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, data_target)
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

        tmp_mae, tmp_rmse, len_less10 = linear3(data_feature, data_target)
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












