# ---- coding: utf-8 ----
# @author: Xing Wei
# @version: v14, demonstration of mixed kernel (Linear+matern3+matern5), 
# @license: (C) Copyright 2021, AMML Group Limited.

"""
CIGP, GRP torch model using nn.module
fixed beta
NOTE THIS:
this version uses `Matern3 Kernel`
"""
import os
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


class CIGP(nn.Module):

    def __init__(
            self,
            x,
            y,
            normal_y_mode=0,
            **kwargs
    ):
        # normal_y_mode = 0: normalize y by combing all dimension.
        # normal_y_mode = 1: normalize y by each dimension.
        super(CIGP, self).__init__()
        # normalize x independently for each dimension
        self.x_mean = x.mean(0)
        self.x_std = x.std(0)
        self.x = (x - self.x_mean.expand_as(x)) / (self.x_std.expand_as(x) + EPS)

        if normal_y_mode == 0:
            # normalize y all together
            self.y_mean = y.mean()
            self.y_std = y.std()
        elif normal_y_mode == 1:
            # normalize y by each dimension
            self.y_mean = y.mean(0)
            self.y_std = y.std(0)
        elif normal_y_mode == 2:
            self.y_mean = torch.zeros(1)
            self.y_std = torch.ones(1)

        self.y = (y - self.y_mean.expand_as(y)) / (self.y_std.expand_as(y) + EPS)

        # GP hyper-parameters

        # self.log_beta = nn.Parameter(torch.ones(1) * -5)   # a large noise, ard
        self.log_beta = nn.Parameter(torch.ones(1) * -8)   # a large noise, ard
        # self.log_beta = nn.Parameter(torch.ones(1) * -6)   # a large noise, ard

        self.log_length_rbf = nn.Parameter(torch.zeros(x.shape[1]))  # RBF Kernel length
        self.log_coe_rbf = nn.Parameter(torch.zeros(1))   # RBF Kernel coefficient

        self.log_coe_linear = nn.Parameter(torch.zeros(1))  # Linear Kernel coefficient

        self.log_length_matern3 = torch.nn.Parameter(torch.zeros(x.shape[1]))  # Matern3 Kernel length
        self.log_coe_matern3 = torch.nn.Parameter(torch.zeros(1))  # Matern3 Kernel coefficient

        self.log_length_matern5 = torch.nn.Parameter(torch.zeros(x.shape[1]))  # Matern5 Kernel length
        self.log_coe_matern5 = torch.nn.Parameter(torch.zeros(1))  # Matern5 Kernel coefficient

        # debug validation
        if 'x_te' in kwargs and 'y_te' in kwargs:
            self.x_te = kwargs['x_te']
            self.y_te = kwargs['y_te']

    # customized kernel------------------------------------important
    def kernel_customized(self, x1, x2):
        # return self.kernel_matern3(x1, x2) + self.kernel_matern5(x1, x2)
        return self.kernel_matern3(x1, x2) + self.kernel_matern5(x1, x2) + self.kernel_linear(x1, x2)
        # return self.kernel_rbf(x1, x2) + self.kernel_linear(x1, x2)

    def kernel_rbf(self, x1, x2):
        x1 = x1 / self.log_coe_rbf.exp()
        x2 = x2 / self.log_coe_rbf.exp()
        # L2 norm
        x1_norm2 = torch.sum(x1 * x1, dim=1).view(-1, 1)
        x2_norm2 = torch.sum(x2 * x2, dim=1).view(-1, 1)

        k_rbf = -2.0 * x1 @ x2.t() + x1_norm2.expand(x1.size(0), x2.size(0)) + x2_norm2.t().expand(x1.size(0), x2.size(0))
        k_rbf = self.log_coe_rbf.exp() * torch.exp(-0.5 * k_rbf)
        return k_rbf

    def kernel_linear(self, x1, x2):
        k_linear = self.log_coe_linear.exp() * (x1 @ x2.t())
        return k_linear

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

    def forward(self, x_te):
        x_te = (x_te - self.x_mean.expand_as(x_te)) / self.x_std.expand_as(x_te)

        sigma = self.kernel_customized(self.x, self.x) + self.log_beta.exp().pow(-1) * torch.eye(self.x.size(0)) \
            + JITTER * torch.eye(self.x.size(0))

        kx = self.kernel_customized(self.x, x_te)
        L = torch.cholesky(sigma)
        l_inv_kx, _ = torch.triangular_solve(kx, L, upper=False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(self.y, L)  # torch.linalg.cholesky()
        # var_diag = self.log_coe_rbf.exp().expand(n_test, 1) \
        #     - (l_inv_kx**2).sum(dim=0).view(-1, 1)
        var_diag = self.kernel_customized(x_te, x_te).diag().view(-1, 1) - (l_inv_kx**2).sum(dim=0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        mean = mean * self.y_std.expand_as(mean) + self.y_mean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.y_std ** 2
        return mean, var_diag

    def negative_log_likelihood(self):
        y_num, y_dimension = self.y.shape
        sigma = self.kernel_customized(self.x, self.x) + self.log_beta.exp().pow(-1) * torch.eye(
            self.x.size(0)) + JITTER * torch.eye(self.x.size(0))

        L = torch.linalg.cholesky(sigma)
        # option 1 (use this if torch supports)
        gamma,_ = torch.triangular_solve(self.y, L, upper = False)
        # option 2
        # gamma = L.inverse() @ y       # we can use this as an alternative because L is a lower triangular matrix.

        nll = 0.5 * (gamma ** 2).sum() + L.diag().log().sum() * y_dimension \
              + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('iter', i, 'nnl:{:.5f}'.format(loss.item()))

    def train_adam_debug(self, niteration=10, lr=0.1, fig_pth='./MSE.png'):
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        mse_list = []
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
            mse_list.append(self.valid())
        self.plot_validation(niteration, mse_list, fig_pth)

    def train_bfgs(self, niteration=50, lr=0.1):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)
        for i in range(niteration):
            def closure():
                optimizer.zero_grad()
                loss = self.negative_log_likelihood()
                loss.backward()
                print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
                return loss

            optimizer.step(closure)

    def valid(self) -> float:
        # for debug, abort this when published
        y_mean, _ = self(self.x_te)
        mse = mean_squared_error(self.y_te.detach(), y_mean.detach())
        return mse

    @ staticmethod
    def plot_validation(iter_n: int, mse_list: list, fig_pth: str):
        # for debug, abort this when published
        plt.plot(list(range(iter_n)), mse_list, label='MSE', color='navy')
        plt.legend()
        plt.grid()
        plt.gcf().savefig(fig_pth)
        plt.show()
        plt.close('all')

    # TODO: add conjugate gradient method



MAE = []
RMSE = []
LESS10 = 0
result_MAE_plot = []
result_RMSE_plot = []
result_LESS10_plot = []
if __name__ == "__main__":

    # multi output test
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.hstack([torch.sin(xte),
                       torch.cos(xte),
                        xte.tanh()] )

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(32, 3) * 0.2

    print(xte.shape)
    print(yte.shape)
    print(xtr.shape)
    print(ytr.shape)

    model = CIGP(xtr, ytr, 1)
    model.train_adam(200, lr=0.1)
    # model.train_bfgs(50, lr=0.001)

    with torch.no_grad():
        ypred, ystd = model(xte)

    # plt.errorbar(xte, ypred.detach(), ystd.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)
    plt.plot(xte, ypred.detach(),'r-.', label='ypred')
    plt.plot(xtr, ytr, 'b+', label='ytrain')
    plt.plot(xte, yte, 'k-', label='ytest')
    plt.legend()
    plt.show()

    # plt.close('all')
    plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        plt.plot(xte, yte[:, i], label='truth', color='r')
        plt.plot(xte, ypred[:, i], label='prediction', color='navy')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ystd[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ystd[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

# %%================================================================================
#     df1 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b17_timingAnalysis1500x5.csv")
    df1 = pd.read_csv("/机器学习/BETA_Baysian Enhanced_STA/Benchmark/b17_VTLx5.csv")
    df2 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\b18_timingAnalysis7200x5.csv")
    # df3 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\timing9500x14.csv")
    # df4 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\timing20000x14.csv")
    # df5 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\timing50000x14.csv")
    # df6 = pd.read_csv("..\\..\\Benchmark\\Benchmark\\timing100000x14.csv")
    df_data1 = np.array(df1.values[:, 1:])
    df_data2 = np.array(df2.values[:1500, 1:])
    # df_data3 = np.array(df3.values[:1500, 1:])
    # df_data4 = np.array(df4.values[:1500, 1:])
    # df_data5 = np.array(df5.values[:1500, 1:])
    # df_data6 = np.array(df6.values[:1500, 1:])
#--------------------------------------------------------------
    #b17
    list_result_less10 = []
    for i in range(df_data1.shape[1]):
        data_feature = df_data1[:,i].reshape(-1,1)  #第 i 列
        data_target = np.delete(df_data1,i,axis=1)  #del 第 i 列
        for j in data_target.T:  #对 列 进行迭代
            xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
            xtr = torch.Tensor(xtr).view(-1, 1)
            xte = torch.Tensor(xte).view(-1, 1)
            ytr = torch.Tensor(ytr).view(-1, 1)
            yte = torch.Tensor(yte).view(-1, 1)
            model = CIGP(xtr, ytr,1)
            model.train_adam(160, lr=0.1)
            with torch.no_grad():
                ypred, ypred_var = model(xte)
            mae = metrics.mean_absolute_error(yte, ypred)
            rmse = metrics.mean_squared_error(yte, ypred)
            MAE.append(mae)
            RMSE.append(rmse)

            Epsilon = yte.reshape(-1) - ypred.reshape(-1)
            abs_Epsilon = np.maximum(Epsilon, -Epsilon)
            less10 = len(abs_Epsilon[abs_Epsilon < 10])
            LESS10 += less10
            print("testY:", yte.shape, "y_pred", ypred.shape)
            print("abs_Epsilon", abs_Epsilon.shape)
            print("MAE:", mae)
            print("RMSE:", rmse)
            print("the num of less10:", less10)  # 返回的是满足条件的个数
            #plot
            plt.plot(xtr, ytr, 'g+',label='train_set')
            # plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='r-.', alpha=0.2)
            plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='o',color="r", ecolor='hotpink', alpha=0.2,label="test_set")
            plt.xlabel("Cornerx")
            plt.ylabel('Cornery')
            plt.title('timing Predition')
            plt.legend()
            plt.show()

        one_LESS10 = LESS10 / (df_data1.shape[0] * (df_data1.shape[1] - 1) * 0.25)  # 乘以 test_size
        LESS10 = 0  # 每一轮记得清零！
        list_result_less10.append(one_LESS10)
        #break  # 测试 一次
    print("==================================================================")
    print("pridiction iteration:", len(MAE))  # 13*14 次
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
    print("This BenchMark Done.")
    print("==========next starting...=============")
#
#MAE 9.45638015624494e-05
#RMSE 1.7582605935309367e-08
#LESS10: 1.0


#============================================================================
# #--------------------------------------------------------------
#     #b18
#     list_result_less10 = []
#     for i in range(df_data2.shape[1]):
#         data_feature = df_data2[:,i].reshape(-1,1)  #第 i 列
#         data_target = np.delete(df_data2,i,axis=1)  #del 第 i 列
#         for j in data_target.T:  #对 列 进行迭代
#             xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
#             xtr = torch.Tensor(xtr).view(-1, 1)
#             xte = torch.Tensor(xte).view(-1, 1)
#             ytr = torch.Tensor(ytr).view(-1, 1)
#             yte = torch.Tensor(yte).view(-1, 1)
#             model = CIGP(xtr, ytr,1)
#             model.train_adam(160, lr=0.1)
#             with torch.no_grad():
#                 ypred, ypred_var = model(xte)
#             mae = metrics.mean_absolute_error(yte, ypred)
#             rmse = metrics.mean_squared_error(yte, ypred)
#             MAE.append(mae)
#             RMSE.append(rmse)
#
#             Epsilon = yte.reshape(-1) - ypred.reshape(-1)
#             abs_Epsilon = np.maximum(Epsilon, -Epsilon)
#             less10 = len(abs_Epsilon[abs_Epsilon < 10])
#             LESS10 += less10
#             print("testY:", yte.shape, "y_pred", ypred.shape)
#             print("abs_Epsilon", abs_Epsilon.shape)
#             print("MAE:", mae)
#             print("RMSE:", rmse)
#             print("the num of less10:", less10)  # 返回的是满足条件的个数
#         one_LESS10 = LESS10 / (df_data2.shape[0] * (df_data2.shape[1] - 1) * 0.25)  # 乘以 test_size
#         LESS10 = 0  # 每一轮记得清零！
#         list_result_less10.append(one_LESS10)
#         break  # 测试 一次
#     print("==================================================================")
#     print("pridiction iteration:", len(MAE))  # 13*14 次
#     result_mae = sum(MAE) / len(MAE)
#     print("MAE", result_mae)
#     result_rmse = sum(RMSE) / len(RMSE)
#     print("RMSE", result_rmse)
#     result_less10 = sum(list_result_less10) / len(list_result_less10)
#     print("LESS10:", result_less10)
#     result_MAE_plot.append(result_mae)
#     result_RMSE_plot.append(result_rmse)
#     result_LESS10_plot.append(result_less10)
#     MAE.clear()
#     RMSE.clear()
#     result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0
#     print("This BenchMark Done.")
#     print("==========next starting...=============")

#MAE 75.033812889686
#RMSE 7484.51513671875
#LESS10: 0.06174358974358975


# #--------------------------------------------------------------
#     #b19
#     list_result_less10 = []
#     for i in range(df_data3.shape[1]):
#         data_feature = df_data3[:,i].reshape(-1,1)  #第 i 列
#         data_target = np.delete(df_data3,i,axis=1)  #del 第 i 列
#         for j in data_target.T:  #对 列 进行迭代
#             xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
#             xtr = torch.Tensor(xtr).view(-1, 1)
#             xte = torch.Tensor(xte).view(-1, 1)
#             ytr = torch.Tensor(ytr).view(-1, 1)
#             yte = torch.Tensor(yte).view(-1, 1)
#             model = CIGP(xtr, ytr,1)
#             model.train_adam(160, lr=0.1)
#             with torch.no_grad():
#                 ypred, ypred_var = model(xte)
#             mae = metrics.mean_absolute_error(yte, ypred)
#             rmse = metrics.mean_squared_error(yte, ypred)
#             MAE.append(mae)
#             RMSE.append(rmse)
#
#             Epsilon = yte.reshape(-1) - ypred.reshape(-1)
#             abs_Epsilon = np.maximum(Epsilon, -Epsilon)
#             less10 = len(abs_Epsilon[abs_Epsilon < 10])
#             LESS10 += less10
#             print("testY:", yte.shape, "y_pred", ypred.shape)
#             print("abs_Epsilon", abs_Epsilon.shape)
#             print("MAE:", mae)
#             print("RMSE:", rmse)
#             print("the num of less10:", less10)  # 返回的是满足条件的个数
#         one_LESS10 = LESS10 / (df_data3.shape[0] * (df_data3.shape[1] - 1) * 0.25)  # 乘以 test_size
#         LESS10 = 0  # 每一轮记得清零！
#         list_result_less10.append(one_LESS10)
#         break  # 测试 一次
#     print("==================================================================")
#     print("pridiction iteration:", len(MAE))  # 13*14 次
#     result_mae = sum(MAE) / len(MAE)
#     print("MAE", result_mae)
#     result_rmse = sum(RMSE) / len(RMSE)
#     print("RMSE", result_rmse)
#     result_less10 = sum(list_result_less10) / len(list_result_less10)
#     print("LESS10:", result_less10)
#     result_MAE_plot.append(result_mae)
#     result_RMSE_plot.append(result_rmse)
#     result_LESS10_plot.append(result_less10)
#     MAE.clear()
#     RMSE.clear()
#     result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0
#     print("This BenchMark Done.")
#     print("==========next starting...=============")

#MAE 50.278374892014725
#RMSE 3362.770301231971
#LESS10: 0.09887179487179487

# #--------------------------------------------------------------
#     #b20
#     list_result_less10 = []
#     for i in range(df_data4.shape[1]):
#         data_feature = df_data4[:,i].reshape(-1,1)  #第 i 列
#         data_target = np.delete(df_data4,i,axis=1)  #del 第 i 列
#         for j in data_target.T:  #对 列 进行迭代
#             xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
#             xtr = torch.Tensor(xtr).view(-1, 1)
#             xte = torch.Tensor(xte).view(-1, 1)
#             ytr = torch.Tensor(ytr).view(-1, 1)
#             yte = torch.Tensor(yte).view(-1, 1)
#             model = CIGP(xtr, ytr,1)
#             model.train_adam(160, lr=0.1)
#             with torch.no_grad():
#                 ypred, ypred_var = model(xte)
#             mae = metrics.mean_absolute_error(yte, ypred)
#             rmse = metrics.mean_squared_error(yte, ypred)
#             MAE.append(mae)
#             RMSE.append(rmse)
#
#             Epsilon = yte.reshape(-1) - ypred.reshape(-1)
#             abs_Epsilon = np.maximum(Epsilon, -Epsilon)
#             less10 = len(abs_Epsilon[abs_Epsilon < 10])
#             LESS10 += less10
#             print("testY:", yte.shape, "y_pred", ypred.shape)
#             print("abs_Epsilon", abs_Epsilon.shape)
#             print("MAE:", mae)
#             print("RMSE:", rmse)
#             print("the num of less10:", less10)  # 返回的是满足条件的个数
#         one_LESS10 = LESS10 / (df_data4.shape[0] * (df_data4.shape[1] - 1) * 0.25)  # 乘以 test_size
#         LESS10 = 0  # 每一轮记得清零！
#         list_result_less10.append(one_LESS10)
#         break  # 测试 一次
#     print("==================================================================")
#     print("pridiction iteration:", len(MAE))  # 13*14 次
#     result_mae = sum(MAE) / len(MAE)
#     print("MAE", result_mae)
#     result_rmse = sum(RMSE) / len(RMSE)
#     print("RMSE", result_rmse)
#     result_less10 = sum(list_result_less10) / len(list_result_less10)
#     print("LESS10:", result_less10)
#     result_MAE_plot.append(result_mae)
#     result_RMSE_plot.append(result_rmse)
#     result_LESS10_plot.append(result_less10)
#     MAE.clear()
#     RMSE.clear()
#     result_mae, result_rmse, result_less10, LESS10 = 0, 0, 0, 0
#     print("This BenchMark Done.")
#     print("==========next starting...=============")
#MAE 74.95469841590294
#RMSE 7518.544921875
#LESS10: 0.06892307692307692


#--------------------------------------------------------------
    # #b21
    # list_result_less10 = []
    # for i in range(df_data5.shape[1]):
    #     data_feature = df_data5[:,i].reshape(-1,1)  #第 i 列
    #     data_target = np.delete(df_data5,i,axis=1)  #del 第 i 列
    #     for j in data_target.T:  #对 列 进行迭代
    #         xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
    #         xtr = torch.Tensor(xtr).view(-1, 1)
    #         xte = torch.Tensor(xte).view(-1, 1)
    #         ytr = torch.Tensor(ytr).view(-1, 1)
    #         yte = torch.Tensor(yte).view(-1, 1)
    #         model = CIGP(xtr, ytr,1)
    #         model.train_adam(160, lr=0.1)
    #         with torch.no_grad():
    #             ypred, ypred_var = model(xte)
    #         mae = metrics.mean_absolute_error(yte, ypred)
    #         rmse = metrics.mean_squared_error(yte, ypred)
    #         MAE.append(mae)
    #         RMSE.append(rmse)
    #
    #         Epsilon = yte.reshape(-1) - ypred.reshape(-1)
    #         abs_Epsilon = np.maximum(Epsilon, -Epsilon)
    #         less10 = len(abs_Epsilon[abs_Epsilon < 10])
    #         LESS10 += less10
    #         print("testY:", yte.shape, "y_pred", ypred.shape)
    #         print("abs_Epsilon", abs_Epsilon.shape)
    #         print("MAE:", mae)
    #         print("RMSE:", rmse)
    #         print("the num of less10:", less10)  # 返回的是满足条件的个数
    #     one_LESS10 = LESS10 / (df_data5.shape[0] * (df_data5.shape[1] - 1) * 0.25)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     break  # 测试 一次
    # print("==================================================================")
    # print("pridiction iteration:", len(MAE))  # 13*14 次
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
    # print("This BenchMark Done.")
    # print("==========next starting...=============")

# MAE 380.9361548790565
# RMSE 191919.38221153847
# LESS10: 0.013743589743589744

#---------------------------------------------------
    # #b22
    # list_result_less10 = []
    # for i in range(df_data6.shape[1]):
    #     data_feature = df_data6[:,i].reshape(-1,1)  #第 i 列
    #     data_target = np.delete(df_data6,i,axis=1)  #del 第 i 列
    #     for j in data_target.T:  #对 列 进行迭代
    #         xtr, xte, ytr, yte = train_test_split(data_feature, j.reshape(-1,1), test_size=0.25)
    #         xtr = torch.Tensor(xtr).view(-1, 1)
    #         xte = torch.Tensor(xte).view(-1, 1)
    #         ytr = torch.Tensor(ytr).view(-1, 1)
    #         yte = torch.Tensor(yte).view(-1, 1)
    #         model = CIGP(xtr, ytr,1)
    #         model.train_adam(160, lr=0.1)
    #         with torch.no_grad():
    #             ypred, ypred_var = model(xte)
    #         mae = metrics.mean_absolute_error(yte, ypred)
    #         rmse = metrics.mean_squared_error(yte, ypred)
    #         MAE.append(mae)
    #         RMSE.append(rmse)
    #
    #         Epsilon = yte.reshape(-1) - ypred.reshape(-1)
    #         abs_Epsilon = np.maximum(Epsilon, -Epsilon)
    #         less10 = len(abs_Epsilon[abs_Epsilon < 10])
    #         LESS10 += less10
    #         print("testY:", yte.shape, "y_pred", ypred.shape)
    #         print("abs_Epsilon", abs_Epsilon.shape)
    #         print("MAE:", mae)
    #         print("RMSE:", rmse)
    #         print("the num of less10:", less10)  # 返回的是满足条件的个数
    #     one_LESS10 = LESS10 / (df_data6.shape[0] * (df_data6.shape[1] - 1) * 0.25)  # 乘以 test_size
    #     LESS10 = 0  # 每一轮记得清零！
    #     list_result_less10.append(one_LESS10)
    #     break  # 测试 一次
    # print("==================================================================")
    # print("pridiction iteration:", len(MAE))  # 13*14 次
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
    # print("This BenchMark Done.")
    # print("==========next starting...=============")

    # MAE
    # 99.71036822979266
    # RMSE
    # 13257.560471754809
    # LESS10: 0.05046153846153846