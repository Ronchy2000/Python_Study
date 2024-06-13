# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 17:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : RF_transfer_15nm_2idea.py
# @Software: PyCharm


#This test based on b18_v12345
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
values = ['Corner1','Corner2','Corner3','Corner4','Corner5']



def RF_regression(data_feature,data_target):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_feature, data_target,
                                                    test_size = test_size)  # 25% 作为测试集
    # model = RandomForestRegressor()
    model = RandomForestRegressor(n_estimators=10,max_depth=4,bootstrap=True)



    model.fit(Xtrain, Ytrain)
    ypred = model.predict(Xtest)
    rmse = mean_squared_error(Ytest, ypred)
    mae = mean_absolute_error(Ytest, ypred)

    Epsilon = Ytest.reshape(-1) - ypred.reshape(-1)
    abs_Epsilon = np.maximum(Epsilon, -Epsilon)

    less10 = len(abs_Epsilon[abs_Epsilon < 10])
    print("testY:", Ytest.shape, "y_pred", ypred.shape)
    print("abs_Epsilon", abs_Epsilon.shape)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("the num of less10:", less10)  # 返回的是满足条件的个数
    return mae, rmse, less10

MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = np.empty(shape = [1,5])
result_RMSE_plot = np.empty(shape = [1,5])
result_LESS10_plot = np.empty(shape = [1,5])
if __name__ == "__main__":

    df1 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b17_15nm_5次迭代//b17_15nm_v1_x5.csv")
    df2 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b17_15nm_5次迭代//b17_15nm_v2_x5.csv")
    df3 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b17_15nm_5次迭代//b17_15nm_v3_x5.csv")
    df4 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b17_15nm_5次迭代//b17_15nm_v4_x5.csv")
    df5 = pd.read_csv("../../../Benchmark/Benchmark/15nm//b17_15nm_5次迭代//b17_15nm_v5_x5.csv")

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

        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, data_target)
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

        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, data_target)
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
    result_LESS10_plot = np.concatenate((result_LESS10_plot.reshape(1, -1), np.array(list_result_less10).reshape(1, -1)), axis=0)

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

        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, data_target)
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

        tmp_mae, tmp_rmse, len_less10 = RF_regression(data_feature, data_target)
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


#plot
    x = [i for i in range(1, len(MAE) + 1)]
    print(result_MAE_plot.shape)
    plt.figure(1)
    plt.plot(x, result_MAE_plot[0, :], label='one_generation')
    plt.plot(x, result_MAE_plot[1, :], label='two_generation')
    plt.plot(x, result_MAE_plot[2, :], label='three_generation')
    plt.plot(x, result_MAE_plot[3, :], label='four_generation')
    plt.xticks(x, values)
    plt.title("MAE")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(x, result_RMSE_plot[0, :], label='one_generation')
    plt.plot(x, result_RMSE_plot[1, :], label='two_generation')
    plt.plot(x, result_RMSE_plot[2, :], label='three_generation')
    plt.plot(x, result_RMSE_plot[3, :], label='four_generation')
    plt.xticks(x, values)
    plt.legend()
    plt.title("RMSE")
    plt.show()

    plt.figure(3)
    plt.plot(x, result_LESS10_plot[0, :]*100, label='one_generation')
    plt.plot(x, result_LESS10_plot[1, :]*100, label='two_generation')
    plt.plot(x, result_LESS10_plot[2, :]*100, label='three_generation')
    plt.plot(x, result_LESS10_plot[3, :]*100, label='four_generation')
    plt.xticks(x, values)
    plt.legend()
    plt.title("LESS10")
    plt.show()


'''
result_MAE_plot= [[18.34935915 18.16733671 17.49153884 13.60220931 19.45993397]
 [15.41092666 16.57298156 16.14532966 15.35908606 21.05127144]
 [14.99833058 18.30046546 16.39100414 15.59113691 18.36272365]
 [15.25504747 20.08981353 16.01489362 16.39390791 22.19627534]]
result_RMSE_plot= [[529.84273933 480.25735813 457.86367398 291.01392289 579.73658433]
 [351.12419404 418.22235311 375.40941306 351.11483535 639.54385359]
 [355.68123972 585.16766668 408.01324282 380.00118283 529.05668817]
 [360.14051646 641.67794444 381.11425704 374.89214684 770.78400647]]
result_LESS10_plot= [[0.82314815 0.8462963  0.85740741 0.91944444 0.78425926]
 [0.90925926 0.89259259 0.90925926 0.86481481 0.72962963]
 [0.88333333 0.82962963 0.875      0.90277778 0.79814815]
 [0.88055556 0.83055556 0.87592593 0.88888889 0.73055556]]



'''