# -*- coding: utf-8 -*-
# @Time    : 2022/8/24 10:46
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : HelloGP.py
# @Software: PyCharm

# https://github.com/AMMLE/EasyGP/blob/main/sgp_chs.ipynb

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
# from torch.autograd import Variable
import pandas as pd


print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415
#生成综合数据
#train_set
# xtr = torch.rand(32, 1)
# print(xtr,xtr.shape)
# ytr = ((6*xtr - 2)**2) * torch.sin(12*xtr - 4) + torch.randn(32, 1) * 1
# print(ytr,ytr.shape)

df = pd.read_csv("timing1500x14.csv")
# x_path_tr = [i for i in range( df.shape[0]) ]
# y_corner_tr = np.array(df.values[:,1])
xtr= np.array( [i for i in range( df.shape[0])]).reshape(-1,1)
xtr = torch.from_numpy(xtr)
xtr = xtr.double()

ytr= np.array(df.values[:,1]).reshape(-1,1)
ytr = torch.from_numpy(ytr)
ytr = ytr.double()
print(ytr,ytr.shape)
#test_set
# xte = torch.linspace(0, 1, 100).view(-1,1)
# yte = ((6*xte - 2)**2) * torch.sin(12*xte - 4)
xte = np.array([i for i in range( df.shape[0]) ]).reshape(-1,1)
xte = torch.from_numpy(xte)
xte = xte.double()

yte = np.array(df.values[:,2]).reshape(-1,1)
yte = torch.from_numpy(yte)
yte = yte.double()
# x_path_te = [i for i in range( df.shape[0]) ]
# y_corner_te = np.array(df.values[:,2])

#plot the data
print("xtr.size:", xtr.size(), "ytr.size:", ytr.size())
print("xte.size:", xte.size(), "yte.size:", yte.size())
plt.plot(xtr.numpy(), ytr.numpy(), 'b+')
plt.plot(xte.numpy(), yte.numpy(), 'r-', alpha = 0.5)
plt.show()

# define kernel parameters
log_length_scale = nn.Parameter(torch.zeros(xte.size(1)))
log_scale = nn.Parameter(torch.zeros(1))
def kernel(X1, X2, log_length_scale, log_scale): # 定义核函数没有加linear

    X1 = X1 / log_length_scale.exp()**2
    X2 = X2 / log_length_scale.exp()**2

    X1_norm2 = X1 * X1
    X2_norm2 = X2 * X2

    K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
    K = log_scale.exp() * torch.exp(-0.5 * K)
    return K

K2 = kernel(xtr, xtr, log_length_scale, log_scale)
print(K2)
# print( (K1 - K2).norm() )

log_beta = nn.Parameter(torch.ones(1) * -4) # this is a large noise. we optimize to shrink it to a proper value.


def negative_log_likelihood(X, Y, log_length_scale, log_scale, log_beta):
    y_num = Y.size(0)
    Sigma = kernel(X, X, log_length_scale, log_scale) + log_beta.exp().pow(-1) * torch.eye(
        X.size(0)) + JITTER * torch.eye(X.size(0))  # add JITTER here to avoid singularity

    L = torch.linalg.cholesky(Sigma)
    # option 1 (use this if torch supports)
    gamma, _ = torch.triangular_solve(Y, L, upper=False)
    # option 2
    # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

    nll = 0.5 * (gamma ** 2).sum() + L.diag().log().sum() + 0.5 * y_num * torch.log(2 * torch.tensor(PI))
    return nll


def train_adam(X, Y, log_length_scale, log_scale, log_beta, niter=50, lr=0.001):
    optimizer = torch.optim.Adam([log_beta, log_length_scale, log_scale], lr=lr)
    optimizer.zero_grad()
    for i in range(niter):
        optimizer.zero_grad()
        # self.update()
        loss = negative_log_likelihood(X, Y, log_length_scale, log_scale, log_beta)
        loss.backward()
        optimizer.step()

        # print the nll
        # print('iter', i, ' nnl:', loss.item())
        print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
        # print the likelihood
        # print('iter', i, 'nnl:{:.5f}'.format(loss.item()),'likelihood:{:.9f}'.format((-loss).exp().item()) )


train_adam(xtr, ytr, log_length_scale, log_scale, log_beta, 15, 0.01)


def forward(X, Xte, log_length_scale, log_scale, log_beta, Y):
    n_test = Xte.size(0)
    Sigma = kernel(X, X, log_length_scale, log_scale) + log_beta.exp().pow(-1) * torch.eye(
        X.size(0)) + JITTER * torch.eye(X.size(0))
    kx = kernel(X, Xte, log_length_scale, log_scale)
    L = torch.cholesky(Sigma)

    # option 1
    mean = kx.t() @ torch.cholesky_solve(Y, L)  # torch.linalg.cholesky()
    # option 2
    # mean = kx @ torch.L.t().inverse() @ L.inverse() @ Y

    # LinvKx = L.inverse() @ kx.t()  # TODO: the inverse for L should be cheap. check this.
    # torch.cholesky_solve(kx.t(), L)
    LinvKx, _ = torch.triangular_solve(kx, L, upper=False)
    # option 1, standard way
    # var_diag = kernel(Xte, Xte, log_length_scale, log_scale).diag().view(-1,1) - (LinvKx.t() @ LinvKx).diag().view(-1, 1)
    # option 2, a faster way
    var_diag = log_scale.exp().expand(n_test, 1) - (LinvKx ** 2).sum(dim=0).view(-1, 1)

    var_diag = var_diag + log_beta.exp().pow(-1)
    return mean, var_diag


train_adam(xtr, ytr, log_length_scale, log_scale, log_beta, 50, 0.1)
with torch.no_grad():
    ypred, yvar = forward(xtr, xte, log_length_scale, log_scale, log_beta, ytr)

# plt.errorbar(xte.numpy().reshape(100), ypred.detach().numpy().reshape(100),
#              yerr=yvar.sqrt().squeeze().detach().numpy(), fmt='r-.', alpha=0.2)
plt.errorbar(xte.numpy().reshape(1500), ypred.detach().numpy().reshape(1500),
             yerr=yvar.sqrt().squeeze().detach().numpy(), fmt='r-.', alpha=0.2)
plt.plot(xtr.numpy(), ytr.numpy(), 'b+')
plt.show()