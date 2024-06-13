# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 19:33
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : GPR_sklearn.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from math import *
from matplotlib import pyplot as  plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C,RationalQuadratic as RQ,WhiteKernel,ExpSineSquared as Exp
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn import metrics


df = pd.read_csv("timing1500x14.csv")
df_array = np.array(df)
Corner1 = df_array[:,1].reshape(-1,1)
Corner2 = df_array[:,2].reshape(-1,1)
xtr,xte,ytr,yte = train_test_split(Corner1,Corner2,test_size = 0.3)

#--------------------------------------------------
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
# length_scale_bounds=(1e-05,2), alpha_bounds=(1e-05,100000.0) + Exp(length_scale=1,periodicity=1)

model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

model.fit(xtr,ytr)

y_pred,sigma_1 = model.predict(xte,return_std=True)
# print("y_pred:",y_pred)
# print("sigma_1:",sigma_1)
#--------------------------------------------------
MAE = metrics.mean_absolute_error(yte, y_pred)
print("MAE:",MAE)


xte = xte.reshape(-1)
y_pred = y_pred.reshape(-1)
print(xte.shape,y_pred.shape,sigma_1.shape)
# plt.plot(xtr, ytr, 'b+')
plt.errorbar(xte, y_pred, yerr = np.sqrt(sigma_1), fmt='r-.', alpha=0.2)
plt.xlabel("Corner1")
plt.ylabel('Corner2')
plt.show()
#plot figure







