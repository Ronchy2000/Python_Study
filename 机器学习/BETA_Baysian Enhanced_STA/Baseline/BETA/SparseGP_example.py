# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 13:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : SparseGP_example.py
# @Software: PyCharm

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

import urllib.request
import os
from scipy.io import loadmat
from math import floor


# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
print("smoke_test:",smoke_test)

if not smoke_test and not os.path.isfile('../elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(1000, 3), torch.randn(1000)  #用不到
else:
    ## 该例程用到的是这个
    data = torch.Tensor(loadmat('../elevators.mat')['data'])
    X = data[:, :-1]
    print("X处理前:",X)
    #数据处理 ，不太懂
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    print("X处理后:", X)
    y = data[:, -1]
    # feature:18
    # target: 1
    print("X.size:",X.size(),"\ny.size:",y.size())

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()  # contiguous 类似于深拷贝
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

print("wheather cuda is available? ",torch.cuda.is_available())
#False  跳过以下代码，变成
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

#
# ##  Defining the SGPR Model
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

#
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# # Training the model
training_iterations = 2 if smoke_test else 50

## Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

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

# 模型训练
train()


#预测
model.eval()
likelihood.eval()
with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    preds = model(test_x)
print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))








