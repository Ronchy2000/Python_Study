# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 16:26
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : plot_diff_kernal.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

#Kernal_RBF
RBF_result_MAE_plot =np.sqrt( np.array([0.21049794669334704, 76.15701293945312, 50.05299758911133, 75.49973414494441, 376.6522146371695, 99.56757530799278]))/100
RBF_result_RMSE_plot =np.sqrt( np.array([0.061572681252772994, 7678.154259314904, 3338.954383263221, 7596.902118389423, 188439.15865384616, 13252.0126953125]))/100
RBF_result_LESS10_plot =np.array([0.05928205128205128, 0.10112820512820513, 0.06646153846153846, 0.012307692307692308, 0.04923076923076923])*100

#Kernal:  matern3+5+linear
matern3_5_linear_result_MAE_plot = np.sqrt( np.array([9.45638015624494e-05, 75.033812889686, 50.278374892014725, 74.95469841590294,380.9361548790565,99.71036822979266]))/100
matern3_5_linear_result_RMSE_plot = np.sqrt( np.array([1.7582605935309367e-08, 7484.51513671875, 3362.770301231971, 7518.544921875, 191919.38221153847,13257.560471754809]))/100
matern3_5_linear_result_LESS10_plot = np.array([0.06174358974358975, 0.09887179487179487,0.06892307692307692, 0.013743589743589744,0.05046153846153846])*100

##figure - MAE
plt.figure(1)
x_ax = range(1, len(RBF_result_MAE_plot) + 1)
plt.plot(x_ax, RBF_result_MAE_plot, 'orangered', linewidth=1, label="sigle_RBF")
plt.plot(x_ax, matern3_5_linear_result_MAE_plot, 'lawngreen', linewidth=1, label="multi_kernel")
plt.title("MAE")
plt.xlabel('benchmark')
plt.ylabel('MAE_value(ps)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(0) #不显示网格线
plt.show()

plt.figure(2)
# x_ax = range(1, len(RBF_result_MAE_plot) + 1)
plt.plot(x_ax, RBF_result_RMSE_plot, 'orangered', linewidth=1, label="sigle_RBF")
plt.plot(x_ax, matern3_5_linear_result_RMSE_plot, 'lawngreen', linewidth=1, label="multi_kernel")
plt.title("RMSE")
plt.xlabel('benchmark')
plt.ylabel('RMSE_value(ps)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(0) #不显示网格线
plt.show()

plt.figure(3)
x_ax = range(1, len(RBF_result_LESS10_plot) + 1)
plt.plot(x_ax, RBF_result_LESS10_plot, 'orangered', linewidth=1, label="sigle_RBF")
plt.plot(x_ax, matern3_5_linear_result_LESS10_plot, 'lawngreen', linewidth=1, label="multi_kernel")
plt.title("LESS10")
plt.xlabel('benchmark')
plt.ylabel('LESS10(%)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(0) #不显示网格线
plt.show()
