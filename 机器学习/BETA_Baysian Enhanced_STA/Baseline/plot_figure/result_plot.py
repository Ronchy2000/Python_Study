# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


values = ['b17_v1', 'b17_v2', 'b17_v3', 'b18', 'b19']

##Figure MAE
Ridge_result_MAE_plot = [111.30132809817754, 57.90079059731588, 17.273904424443664, 157.32547192624403, 125.82300093776868]
MLP_result_MAE_plot = [98.01578503469811, 55.049662424799976, 16.55337208986829, 140.08156517100534, 114.52690422859027]
RF_result_MAE_plot = [44.1975537117615, 45.79078616768285, 15.110304415147775, 38.31763357169766, 80.46698514996518]
GP_result_MAE_plot = [76.34740829467773,26.090026378631592,16.58118987083435,123.9825439453125,109.82885932922363]

plt.figure(1)
#plot
x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_MAE_plot, color="green", marker='o',markersize = 5, linewidth=0.7, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_MAE_plot, color="blue", marker='v', markersize = 5, linewidth=0.7, label="MLP")
#RF
plt.plot(x_ax, RF_result_MAE_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")
#GP
plt.plot(x_ax,GP_result_MAE_plot,'mD-',markersize = 5, linewidth=0.7, label="GP")

plt.ylabel('MAE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.xticks(x_ax, values)
plt.show()
fig1_file = "MAE_plot.eps"
plt.savefig(fig1_file,  bbox_inches='tight')


##Figure RMSE
plt.figure(2)
Ridge_result_RMSE_plot = np.sqrt( np.array([20960.882955096586, 9982.765240790297, 638.8560937249046, 45542.85606003766, 27793.15565676238]) )
MLP_result_RMSE_plot =np.sqrt(np.array([18005.63240078954, 9218.374012075514, 557.7368960290847, 36385.6816289802, 23083.026416110777]))
RF_result_RMSE_plot = np.sqrt(np.array([7504.8215884886085, 7899.028836047025, 509.63355439615145, 7648.482921695517, 15659.565242485802]))
GP_result_RMSE_plot = np.sqrt(np.array([11741.055786132812,1735.0331420898438,608.8122940063477,31130.89794921875,21086.615966796875]))
#plot
# x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_RMSE_plot, color="green", marker='o',markersize = 5, linewidth=0.7, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_RMSE_plot, color="blue", marker='v', markersize = 5, linewidth=0.7, label="MLP")
#RF
plt.plot(x_ax, RF_result_RMSE_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")
#GP
plt.plot(x_ax,GP_result_RMSE_plot,'mD-',markersize = 5, linewidth=0.7, label="GP")


plt.ylabel('RMSE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.xticks(x_ax, values)
plt.show()
fig2_file = "RMSE_plot.eps"
plt.savefig(fig2_file,  bbox_inches='tight')


####Figure LESS30
plt.figure(3)
Ridge_result_LESS30_plot = np.array([0.1806666666666667, 0.5748, 0.8429333333333332, 0.14961111111111108, 0.19407407407407407])*100
MLP_result_LESS30_plot = np.array([0.23186666666666672, 0.6169333333333333, 0.8822666666666666, 0.18422222222222223, 0.19871604938271606])*100
RF_result_LESS30_plot = np.array([0.6284, 0.6669333333333334, 0.8830666666666668, 0.7053888888888888, 0.3838024691358025])*100
GP_result_LESS30_plot = np.array([0.332,0.7646666666666667,0.874, 0.22981770833333334,0.18387345679012346])*100
#plot
x_ax = range(1, len(Ridge_result_LESS30_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_LESS30_plot, color="green", marker='o',markersize = 5, linewidth=0.7, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_LESS30_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")
#RF
plt.plot(x_ax, RF_result_LESS30_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")
#GP
plt.plot(x_ax,GP_result_LESS30_plot,'mD-',markersize = 5, linewidth=0.7, label="GP")



plt.ylabel('LESS30(%)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="upper right")  #set legend location
plt.grid(0)
plt.xticks(x_ax, values)
plt.show()
fig3_file = "LESS30_plot.eps"
plt.savefig(fig3_file,  bbox_inches='tight')
