# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 20:59
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MLP_Regressor2.py
# @Software: PyCharm
# A good tutorial: https://michael-fuchs-python.netlify.app/2021/02/10/nn-multi-layer-perceptron-regressor-mlpregressor/

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

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

    # df_temp = pd.DataFrame({'Actual': testY, 'Predicted': y_pred})
    # df_temp.head()
    #
    # df_temp = df_temp.head(30)
    # df_temp.plot(kind='bar',figsize=(10,6))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()
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
if __name__ == "__main__":
    df = pd.read_csv("timing1500x14.csv")
    df_data = np.array(df.values[:, 1:])

    for i in range(df_data.shape[1]):
        data_feature = df_data[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            tmp = myMLPRegressor(data_feature,j.reshape(-1,1))
            MAE.append(tmp)
    result = sum(MAE)/len(MAE)
    print("MAE",MAE,"\nlen(MAE)",len(MAE))  # 13*14 次
    print("result",result)