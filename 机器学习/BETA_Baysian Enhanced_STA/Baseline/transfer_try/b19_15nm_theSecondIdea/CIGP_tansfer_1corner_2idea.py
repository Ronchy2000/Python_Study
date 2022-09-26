# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 16:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : CIGP_tansfer_1corner_2idea.py
# @Software: PyCharm



# %%
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

test_size = 0.25
values = ['Corner1','Corner2','Corner3','Corner4','Corner5']


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
        L = torch.cholesky(Sigma)
        LinvKx, _ = torch.triangular_solve(kx, L, upper=False)

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
        Gamma, _ = torch.triangular_solve(self.Y, L, upper=False)
        # option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

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
            #print('iter', i, 'nll:{:.5f}'.format(loss.item()))

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
#

MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = np.empty(shape = [1,5])
result_RMSE_plot = np.empty(shape = [1,5])
result_LESS10_plot = np.empty(shape = [1,5])
#设置
LESS_value = 20

one_iteration = 1


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

        xtr, xte, ytr, yte = train_test_split(data_feature, data_target, test_size=test_size)
        xtr = torch.Tensor(xtr).view(-1, 1)
        xte = torch.Tensor(xte).view(-1, 1)
        ytr = torch.Tensor(ytr).view(-1, 1)
        yte = torch.Tensor(yte).view(-1, 1)
        model = cigp(xtr, ytr)
        model.train_adam(150, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)
        mae = metrics.mean_absolute_error(yte, ypred)
        rmse = metrics.mean_squared_error(yte, ypred)
        MAE.append(mae)
        RMSE.append(rmse)
        Epsilon = yte.reshape(-1) - ypred.reshape(-1)
        abs_Epsilon = np.maximum(Epsilon, -Epsilon)
        # LESS  #*************************************
        less10 = len(abs_Epsilon[abs_Epsilon < LESS_value])
        LESS10 += less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)

    print('------------------------------------------------------')
    print('Use one version\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:", list_result_less10)

    result_MAE_plot = np.array(MAE)
    result_RMSE_plot = np.array(RMSE)
    result_LESS10_plot = np.array(list_result_less10)



###two version
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:, i].reshape(-1, 1)
        tmp2 = df_data2[:, i].reshape(-1, 1)
        data_feature = np.concatenate((tmp1,tmp2), axis = 1)  # 第 i 列
        data_target = df_data3[:, i].reshape(-1, 1)  # 第 i 列

        xtr, xte, ytr, yte = train_test_split(data_feature, data_target, test_size=test_size)
        xtr = torch.Tensor(xtr)
        xte = torch.Tensor(xte)
        ytr = torch.Tensor(ytr).view(-1, 1)
        yte = torch.Tensor(yte).view(-1, 1)
        model = cigp(xtr, ytr)
        model.train_adam(150, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)
        mae = metrics.mean_absolute_error(yte, ypred)
        rmse = metrics.mean_squared_error(yte, ypred)
        MAE.append(mae)
        RMSE.append(rmse)
        Epsilon = yte.reshape(-1) - ypred.reshape(-1)
        abs_Epsilon = np.maximum(Epsilon, -Epsilon)
        # LESS  #*************************************
        less10 = len(abs_Epsilon[abs_Epsilon < LESS_value])
        LESS10 += less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)

    print('------------------------------------------------------')
    print('Use two version\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:", list_result_less10)

    result_MAE_plot = np.concatenate((result_MAE_plot.reshape(1, -1), np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot.reshape(1, -1), np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate((result_LESS10_plot.reshape(1, -1), np.array(list_result_less10).reshape(1, -1)), axis=0)

###three version
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:, i].reshape(-1, 1)
        tmp2 = df_data2[:, i].reshape(-1, 1)
        tmp3 = df_data3[:, i].reshape(-1, 1)
        data_feature = np.concatenate((tmp1, tmp2, tmp3), axis=1)  # 第 i 列
        data_target = df_data4[:, i].reshape(-1, 1)  # 第 i 列

        xtr, xte, ytr, yte = train_test_split(data_feature, data_target, test_size=test_size)
        xtr = torch.Tensor(xtr)
        xte = torch.Tensor(xte)
        ytr = torch.Tensor(ytr).view(-1, 1)
        yte = torch.Tensor(yte).view(-1, 1)
        model = cigp(xtr, ytr)
        model.train_adam(150, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)
        mae = metrics.mean_absolute_error(yte, ypred)
        rmse = metrics.mean_squared_error(yte, ypred)
        MAE.append(mae)
        RMSE.append(rmse)
        Epsilon = yte.reshape(-1) - ypred.reshape(-1)
        abs_Epsilon = np.maximum(Epsilon, -Epsilon)
        # LESS  #*************************************
        less10 = len(abs_Epsilon[abs_Epsilon < LESS_value])
        LESS10 += less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)

    print('------------------------------------------------------')
    print('Use three version\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:", list_result_less10)

    result_MAE_plot = np.concatenate((result_MAE_plot, np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot, np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate((result_LESS10_plot, np.array(list_result_less10).reshape(1, -1)), axis=0)

###four version
    MAE.clear()
    RMSE.clear()
    list_result_less10.clear()
    for i in range(df_data1.shape[1]):
        tmp1 = df_data1[:, i].reshape(-1, 1)
        tmp2 = df_data2[:, i].reshape(-1, 1)
        tmp3 = df_data3[:, i].reshape(-1, 1)
        tmp4 = df_data3[:, i].reshape(-1, 1)
        data_feature = np.concatenate((tmp1, tmp2, tmp3, tmp4), axis=1)  # 第 i 列
        data_target = df_data5[:, i].reshape(-1, 1)  # 第 i 列

        xtr, xte, ytr, yte = train_test_split(data_feature, data_target, test_size=test_size)
        xtr = torch.Tensor(xtr)
        xte = torch.Tensor(xte)
        ytr = torch.Tensor(ytr).view(-1, 1)
        yte = torch.Tensor(yte).view(-1, 1)
        model = cigp(xtr, ytr)
        model.train_adam(150, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)
        mae = metrics.mean_absolute_error(yte, ypred)
        rmse = metrics.mean_squared_error(yte, ypred)
        MAE.append(mae)
        RMSE.append(rmse)
        Epsilon = yte.reshape(-1) - ypred.reshape(-1)
        abs_Epsilon = np.maximum(Epsilon, -Epsilon)
        # LESS  #*************************************
        less10 = len(abs_Epsilon[abs_Epsilon < LESS_value])
        LESS10 += less10
        one_LESS10 = LESS10 / (data_target.shape[0] * test_size)
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)

    print('------------------------------------------------------')
    print('Use four version\n')
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("less10:", list_result_less10)

    result_MAE_plot = np.concatenate((result_MAE_plot, np.array(MAE).reshape(1, -1)), axis=0)
    result_RMSE_plot = np.concatenate((result_RMSE_plot, np.array(RMSE).reshape(1, -1)), axis=0)
    result_LESS10_plot = np.concatenate((result_LESS10_plot, np.array(list_result_less10).reshape(1, -1)), axis=0)

    print("======================================================")
    print("result_MAE_plot=", result_MAE_plot)
    print("result_RMSE_plot=", result_RMSE_plot)
    print("result_LESS10_plot=", result_LESS10_plot)
    print("LESS_value",LESS_value)
##plot
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
    plt.plot(x, result_LESS10_plot[0, :] * 100, label='one_generation')
    plt.plot(x, result_LESS10_plot[1, :] * 100, label='two_generation')
    plt.plot(x, result_LESS10_plot[2, :] * 100, label='three_generation')
    plt.plot(x, result_LESS10_plot[3, :] * 100, label='four_generation')
    plt.xticks(x, values)
    plt.legend()
    plt.title("LESS10")
    plt.show()


