# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 19:33
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : GPR_sklearn.py
# @Software: PyCharm

#https://www.youtube.com/watch?v=QvcHrwXS4_U

import numpy as np
import pandas as pd
from math import *
from matplotlib import pyplot as  plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C,RationalQuadratic as RQ,WhiteKernel,ExpSineSquared as Exp
from sklearn.model_selection import train_test_split


df = pd.read_csv("timing1500x14.csv")
df_array = np.array(df)
Corner1 = df_array[:,1].reshape(-1,1)
Corner2 = df_array[:,2].reshape(-1,1)
xtr,xte,ytr,yte = train_test_split(Corner1,Corner2,test_size = 0.3)

#--------------------------------------------------
kernel = C()*Exp(length_scale = 24,periodicity = 1)
# length_scale_bounds=(1e-05,2), alpha_bounds=(1e-05,100000.0) + Exp(length_scale=1,periodicity=1)

gp = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer=4)

gp.fit(xtr,ytr)

y_pred,sigma_1 = gp.predict(xte,return_std=True)
#--------------------------------------------------

fig = plt.figure(num=1,figsize=(11,0.8),dpi=300,facecolor='w',edgecolor='k')
fig.text(0.5,-1,'$Time\[hours]$',ha = 'center')
fig.text(0.04,10,'$Globa\horizontal\irrandiance\[W/m^2]$',va = 'center',rotation = 'vertical')
# plt.subplot(4,1,1)
plt.plot(xtr,ytr,'b.',markersize=5,label=u'obervation')
plt.plot(xte,y_pred,'b-',linewidth =1,label=u'Prediction')
# plt.fill_between()
plt.xlabel('(a)')
plt.legend(loc='upper right',fontsize=10)
plt.ylim(-750,1750)






