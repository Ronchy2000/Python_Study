# -*- coding: utf-8 -*-
# @Time    : 2022/9/22 19:38
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : CIGP_transfer_3corner.py
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
values = ['b19_v2', 'b19_v3','b19_v4']


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



MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
#设置
LESS_value = 30


one_Corner = 1,2,3

if __name__ == "__main__":

    df1 = pd.read_csv("../../Benchmark/Benchmark/15nm//b17_15nm_v1_x5.csv")
    df2 = pd.read_csv("../../Benchmark/Benchmark/15nm//b17_15nm_v2_x5.csv")
    df3 = pd.read_csv("../../Benchmark/Benchmark/15nm//b17_15nm_v3_x5.csv")
    df4 = pd.read_csv("../../Benchmark/Benchmark/15nm//b17_15nm_v4_x5.csv")
    df5 = pd.read_csv("../../Benchmark/Benchmark/15nm//b17_15nm_v5_x5.csv")

    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:, 1:])
    df_data3 = np.array(df3.values[:, 1:])
    df_data4 = np.array(df4.values[:, 1:])
    df_data5 = np.array(df5.values[:, 1:])

    total_data = np.concatenate((df_data1, df_data2, df_data3), axis=0)

    print(total_data.shape)

    # --------------------------------------------------------------
    # #b17_1,b17_2  input   b17_3
    list_result_less10 = []
    Covariance_pred = np.arange(0,1500).reshape(-1,1)
    mean_predict = np.arange(0,1500).reshape(-1,1)

    print("Covariance_pred.shape:",Covariance_pred.shape)

    data_feature = total_data[:,one_Corner]
    data_target = np.delete(total_data, one_Corner, axis=1)  # del 第 1 列

    for j in data_target.T:  #对 列 进行迭代
        # xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=test_size)
        xtr = data_feature[:df_data1.shape[0]*2]
        ytr = j[:df_data1.shape[0]*2].reshape(-1,1)
        xte = data_feature[df_data1.shape[0]*2 : ]
        yte = j[df_data1.shape[0]*2 :].reshape(-1,1)
        # print(xtr.shape,xte.shape,ytr.shape,yte.shape)
        xtr = torch.Tensor(xtr)
        xte = torch.Tensor(xte)
        ytr = torch.Tensor(ytr).view(-1, 1)
        yte = torch.Tensor(yte).view(-1, 1)
        print(xtr.shape, xte.shape, ytr.shape, yte.shape)
        model = cigp(xtr, ytr)
        model.train_adam(150, lr=0.03)
        with torch.no_grad():
            ypred, ypred_var = model(xte)
        # 保存方差
        print("ypred_var.shape",ypred_var.shape)
        Covariance_pred = np.concatenate((Covariance_pred,ypred_var), axis=1) #按列拼接
        mean_predict = np.concatenate((mean_predict,ypred), axis=1) #按列拼接

        mae = metrics.mean_absolute_error(yte, ypred)
        rmse = metrics.mean_squared_error(yte, ypred)
        MAE.append(mae)
        RMSE.append(rmse)
        Epsilon = yte.reshape(-1) - ypred.reshape(-1)
        abs_Epsilon = np.maximum(Epsilon, -Epsilon)
    #LESS  #*************************************
        less10 = len(abs_Epsilon[abs_Epsilon < LESS_value])
        LESS10 += less10
        print("testY:", yte.shape, "y_pred", ypred.shape)
        print("abs_Epsilon", abs_Epsilon.shape)
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("the num of less10:", less10)  # 返回的是满足条件的个数

    one_LESS10 = LESS10 / (df_data1.shape[0] * (df_data1.shape[1] - 1) * test_size)  # 乘以 test_size
    LESS10 = 0  # 每一轮记得清零！
    list_result_less10.append(one_LESS10)
    # dff = pd.DataFrame(Covariance_pred)
    # dff.to_csv("b19_VTL2_covariance.csv", sep=',', index=False)
    # dff2 = pd.DataFrame(mean_predict)
    # dff2.to_csv("b19_VTL2_prediction.csv", sep=',', index=False)
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
    print("b17_v1_15nm BenchMark Done.")
    print("==========next starting...=============")




