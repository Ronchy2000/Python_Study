# -*- coding: utf-8 -*-
# @Time    : 2022/9/16 19:44
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : CIGP_iteration2.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


class cigp(nn.Module):
    def __init__(self, X, Y, normal_y_mode=0):
        # normal_y_mode = 0: normalize Y by combing all dimension.
        # normal_y_mode = 1: normalize Y by each dimension.
        super(cigp, self).__init__()

        # normalize X independently for each dimension
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
        self.X = (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X) + EPS)

        if normal_y_mode == 0:
            # normalize y all together
            self.Ymean = Y.mean()
            self.Ystd = Y.std()
            self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)
        elif normal_y_mode == 1:
            # option 2: normalize y by each dimension
            self.Ymean = Y.mean(0)
            self.Ystd = Y.std(0)
            self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)

        # GP hyperparameters
        self.log_beta = nn.Parameter(
            torch.ones(1) * 0)  # a large noise by default. Smaller value makes larger noise variance.
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))  # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale

    # define kernel function
    def kernel(self, X1, X2):
        # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(
            0))  # this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def kernel_matern3(self, x1, x2):
        """
        latex formula:
        \sigma ^2\left( 1+\frac{\sqrt{3}d}{\rho} \right) \exp \left( -\frac{\sqrt{3}d}{\rho} \right)
        :param x1: x_point1
        :param x2: x_point2
        :return: kernel matrix
        """
        const_sqrt_3 = torch.sqrt(torch.ones(1) * 3)
        x1 = x1 / self.log_length_matern3.exp()
        x2 = x2 / self.log_length_matern3.exp()
        distance = const_sqrt_3 * torch.cdist(x1, x2, p=2)
        k_matern3 = self.log_coe_matern3.exp() * (1 + distance) * (- distance).exp()
        return k_matern3

    def kernel_matern5(self, x1, x2):
        """
        latex formula:
        \sigma ^2\left( 1+\frac{\sqrt{5}}{l}+\frac{5r^2}{3l^2} \right) \exp \left( -\frac{\sqrt{5}distance}{l} \right)
        :param x1: x_point1
        :param x2: x_point2
        :return: kernel matrix
        """
        const_sqrt_5 = torch.sqrt(torch.ones(1) * 5)
        x1 = x1 / self.log_length_matern5.exp()
        x2 = x2 / self.log_length_matern5.exp()
        distance = const_sqrt_5 * torch.cdist(x1, x2, p=2)
        k_matern5 = self.log_coe_matern5.exp() * (1 + distance + distance ** 2 / 3) * (- distance).exp()
        return k_matern5

    def forward(self, Xte):
        n_test = Xte.size(0)
        Xte = (Xte - self.Xmean.expand_as(Xte)) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
                + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(self.X, Xte)
        # L = torch.cholesky(Sigma)
        L = torch.linalg.cholesky(Sigma)
        LinvKx, _ = torch.triangular_solve(kx, L, upper=False)
        # LinvKx, _ = torch.linalg.solve_triangular(kx, L, upper=False)


        # option 1
        mean = kx.t() @ torch.cholesky_solve(self.Y, L)  # torch.linalg.cholesky()

        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) \
                   - (LinvKx ** 2).sum(dim=0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # de-normalized
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.Ystd ** 2

        return mean, var_diag

    def negative_log_likelihood(self):
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))

        L = torch.linalg.cholesky(Sigma)
        # option 1 (use this if torch supports)
        #Gamma, _ = torch.triangular_solve(self.Y, L, upper=False)
        # Gamma, _ = torch.linalg.solve_triangular(self.Y, L, upper=False)
        # option 2
        Gamma = L.inverse() @ self.Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll = 0.5 * (Gamma ** 2).sum() + L.diag().log().sum() * y_dimension \
              + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            # print('loss_nll:', loss.item())
            # print('iter', i, ' nll:', loss.item())
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    def train_bfgs(self, niteration=50, lr=0.1):
        # LBFGS optimizer
        # Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                # self.update()
                loss = self.negative_log_likelihood()
                loss.backward()
                # print('nll:', loss.item())
                # print('iter', i, ' nll:', loss.item())
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)
        # print('loss:', loss.item())

    # TODO: add conjugate gradient method





if __name__ == "__main__":
    # df1 = pd.read_csv("..\\Benchmark\\timing1500x14.csv")
    # df2 = pd.read_csv("..\\Benchmark\\timing3700x14.csv")
    df2 = pd.read_csv("timing3700x14_delete_first_col.csv")
    # df3 = pd.read_csv("..\\Benchmark\\timing9500x14.csv")
    # df4 = pd.read_csv("..\\Benchmark\\timing20000x14.csv")
    # df5 = pd.read_csv("..\\Benchmark\\timing50000x14.csv")
    # df6 = pd.read_csv("..\\Benchmark\\timing100000x14.csv")

# 用前1500行 做实验
#     df_data1 = np.array(df1.values[:, 1:])
#     df_data2 = np.array(df2.values[:1500, 1:])
    df_data2 = df2.loc[:1499,:]
    # df_data3 = np.array(df3.values[:1500, 1:])
    # df_data4 = np.array(df4.values[:1500, 1:])
    # df_data5 = np.array(df5.values[:1500, 1:])
    # df_data6 = np.array(df6.values[:1500, 1:])
    # print("df_data2",df_data2)

    # # 以data2做实验
    seed = 'Corner2'

    feature_name,target_name = [],[]
    header = list(df2.columns.values)  #dataFrame2 的表头
    # Corner2
    seed_index = 1
    ## feature
    feature_name.append(header[seed_index])
    ## target
    target_name = header[:] #拷贝给target_name
    del target_name[seed_index]
    print(df_data2[target_name])

    # feature
    belta = df_data2[feature_name]
    # target
    # gama = df_data2.drop(feature_name, axis=1)
    gama = df_data2[target_name]

    x = np.array(belta)
    y = np.array(gama)
    if x.ndim == 1:  # 如果是一维，就变成二维
        x = x.reshape(-1, 1)  # 1D  -> 2D:针对feature只有一维的时候

    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y[:, 0], test_size=0.25)  # 30% 作为测试集
    #转化为 tensor
    # xtr = torch.Tensor(xtr).view(-1, 1)
    # xte = torch.Tensor(xte).view(-1, 1)
    # ytr = torch.Tensor(ytr).view(-1, 1)
    # yte = torch.Tensor(yte).view(-1, 1)

    MAE, MSE, RMSE = [], [], []
    max_mae, max_mse, max_rmse = [], [], []
    # min_mae, min_mse, min_rmse = [], [], []

    for i in range(y.shape[1]): #获取列
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y[:, i].reshape(-1,1),
                                                                    test_size=0.25)  # 30% 作为测试集
        xtr = torch.Tensor(Xtrain).view(-1, 1)
        xte = torch.Tensor(Xtest).view(-1, 1)
        ytr = torch.Tensor(Ytrain).view(-1, 1)
        yte = torch.Tensor(Ytest).view(-1, 1)

        model = cigp(xtr, ytr)
        model.train_adam(189, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)

        MAE.append( metrics.mean_absolute_error(yte, ypred) )
        RMSE.append(ypred_var.sqrt())
        # print("方差:",ypred_var.sqrt())
        print("MAE:",MAE)

#找最大数
    max_MAE = MAE[0]
    for i in MAE:
        if i > max_MAE:
            max_MAE = i
    print("max_MAE:", max_MAE)
    print("max_MAE.index:", MAE.index(max_MAE))

    max_mae.append(max_MAE)  # domimant 一轮迭代完成 ,plot用

    ###-update feature and target-----------------------
    feature_name.append(target_name[MAE.index(max_MAE)])
    del target_name[MAE.index(max_MAE)]

    MAE.clear()
    print('==========================================================================')
    print("迭代完成")

#-------------
#feature >= 2
    max_iteration = 7
    iteration = 1
    for j in range(max_iteration):
        # iterative process
        print("feature_name:", feature_name)
        print("target_name:", target_name)
        iteration += 1
        print("iteration:", iteration)
        belta = df_data2[feature_name]
        gama = df_data2[target_name]
        x = np.array(belta)
        y = np.array(gama)
        for i in range(y.shape[1]):  # 获取列
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y[:, i].reshape(-1, 1),
                                                            test_size=0.25)  # 30% 作为测试集
            xtr = torch.Tensor(Xtrain)
            xte = torch.Tensor(Xtest)
            ytr = torch.Tensor(Ytrain).view(-1, 1)
            yte = torch.Tensor(Ytest).view(-1, 1)

            model = cigp(xtr, ytr)
            model.train_adam(189, lr=0.03)
            with torch.no_grad():
                ypred, ypred_var = model(xte)

            MAE.append(metrics.mean_absolute_error(yte, ypred))
            RMSE.append(ypred_var.sqrt())
            # print("方差:",ypred_var.sqrt())
            print("MAE:", MAE)
        # 找最大数
        max_MAE = MAE[0]
        for i in MAE:
            if i > max_MAE:
                max_MAE = i
        print("max_MAE:", max_MAE)
        print("max_MAE.index:", MAE.index(max_MAE))
        max_mae.append(max_MAE)  # domimant 一轮迭代完成 ,plot用
        ###-update feature and target-----------------------
        feature_name.append(target_name[MAE.index(max_MAE)])
        del target_name[MAE.index(max_MAE)]
        MAE.clear()
        print('==========================================================================')
        print("迭代完成")



### Plot
    print("max_mae",max_mae)
    x_ax = range(1,len(max_mae)+1)
    plt.plot(x_ax, max_mae, linewidth=1, label="max_mae")
    # plt.plot(x_ax, min_mae, linewidth=1, label="min_mae")
    plt.title("MAE")
    plt.xlabel('the num of dominant Corner')
    plt.ylabel('MAE(ps)')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


