# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 14:38
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : SparseGP_Corner.py
# @Software: PyCharm


import math
import torch
import gpytorch
from matplotlib import pyplot as plt

import urllib.request
import os
from scipy.io import loadmat
from math import floor
import pandas as pd
import numpy as np


# ##  Defining the SGPR Model
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:1500, :], likelihood=likelihood)
        # 1-Dim
        # self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()

if __name__ =="__main__":
    ## 该例程用到的是corner数据集
    df = pd.read_csv("..\\..\\Benchmark\\timing100000x14.csv")
    df_data = np.array(df)

    data = torch.Tensor(df_data)
    # X = data[:, 1:-1]  # 第一列是序号，Corner1 从第二列开始
    X = data[:, [1,2] ]  # 第一列是序号，Corner1 从第二列开始
    print("X处理前:", X)
    # 数据处理 ，不太懂
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    print("X处理后:", X)
    y = data[:, -1]
    # feature:18
    # target: 1
    print("X.size:", X.size(), "\ny.size:", y.size())

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()  # contiguous 类似于深拷贝

    #1-Dim
    # train_x = X[:train_n].contiguous()  # contiguous 类似于深拷贝
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()

    #1-Dim
    # test_x = X[train_n].contiguous()
    test_y = y[train_n:].contiguous()

    #
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    training_iterations = 100
    ## Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # 模型训练
    train()

    #预测
    model.eval()
    likelihood.eval()
    with gpytorch.settings.max_preconditioner_size(1000), torch.no_grad():
        preds = model(test_x)
        print("preds:",preds)
    print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))  #返回值 mean 代表均值






