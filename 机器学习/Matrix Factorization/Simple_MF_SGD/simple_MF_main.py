# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 10:43
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : simple_MF_main.py
# @Software: PyCharm

'''
这种方法最基本的基于SGD的MF方法 ，若是对于矩阵规模大，时间开销很大
不适用
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):  # 矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q = Q.T  # 新生成的Q的转置矩阵               # .T操作表示矩阵的转置
    result = []  # 用于储存加入正则化后的损失函数求和后的值
    for step in range(steps):  # 梯度下降，steps迭代次数
        for i in range(len(R)):  # len(R)代表矩阵的行数
            for j in range(len(R[i])):  # 取每一行的列数
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])  # .DOT表示矩阵相乘
                for k in range(K):
                    if R[i][j] > 0:  # 限制评分大于零  #alpha学习率  ，beta正则化参数
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])  # 增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])  # 增加正则化，并对损失函数求导，然后更新变量Q
        eR = np.dot(P, Q)
        e = 0  # 用来保存损失函数求和后的值
        for i in range(len(R)):  # 每一行循环
            for j in range(len(R[i])):  # 每一列循环
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)  # 损失函数求和
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))  # 加入正则化后的损失函数求和
        result.append(e)
        if e < 0.001:  # 判断是否收敛，0.001为阈值
            break
    return P, Q.T, result


df = pd.read_csv("mydata2_corner1-corner14.csv")
Rin = np.array(df.values[:100,:])
print(Rin)
print(Rin.shape)
if __name__ == '__main__':  # 主函数
    print("MF Start!")
    R = Rin
    # R = np.array(R)
    N = R.shape[0] # 原矩阵R的行数
    M = R.shape[1]  # 原矩阵R的列数
    K = 5  # K值可根据需求改变
    P = np.random.rand(N, K)  # 随机生成一个 N行 K列的矩阵
    Q = np.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵
    nP, nQ, result = matrix_factorization(R, P, Q, K)  # nP=P，nQ=nQ.T,result=result
    print("输出原矩阵：")
    print(R)  # 输出原矩阵
    R_MF = np.dot(nP, nQ.T)  # 矩阵的乘积
    print("输出新矩阵：")
    print(R_MF)  # 输出新矩阵
    MSE = np.square(np.subtract(R,R_MF)).mean()
    rsme = sqrt(MSE)
    print("Root Mean Square Error:",rsme)
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
