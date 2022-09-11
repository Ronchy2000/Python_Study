# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


##Figure MAE
Ridge_result_MAE_plot = [0.12386843565165045, 75.7612244794656, 50.04634842232905, 74.13077984331956, 374.1132485566584, 100.09363098148886]
MLP_result_MAE_plot = [1.82349204425622, 74.72664731060954, 50.28310123823295, 75.0690152183549, 375.0417640192172, 99.93024005355413]
RF_result_MAE_plot = [0.19508149774853237, 86.72016416203127, 57.505882926711124, 86.00809835016133, 432.3321602893996, 112.21760593331265]
GP_result_MAE_plot = [0.19764583,75.17194624680739,45.551545,60.45415155,320.5151564,80.451515]

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

plt.xticks(x_ax, ('b17', 'b18', 'b19', 'b20', 'b21','b22'))
plt.ylabel('MAE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.show()



##Figure RMSE
plt.figure(2)
Ridge_result_RMSE_plot = np.sqrt( np.array([0.02048501388736542, 7618.87465032731, 3338.7829782110844, 7385.211356958652, 186742.10860410004, 13350.491157655804]) )
MLP_result_RMSE_plot =np.sqrt(np.array([5.847694591286665, 7455.867916276916, 3364.560440910698, 7525.512498347117, 187727.88428359214, 13332.76992869297]))
RF_result_RMSE_plot = np.sqrt(np.array([0.07939115520776713, 11014.576478011777, 4812.767409537852, 10831.0829966438, 273429.7573917104, 18226.563149038764]))
GP_result_RMSE_plot = np.sqrt(np.array([0.048697608812718,7001,3000,6000,15000,11000]))
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

plt.xticks(x_ax, ('b17', 'b18', 'b19', 'b20', 'b21','b22'))
plt.ylabel('RMSE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.show()


####Figure LESS10
plt.figure(3)
Ridge_result_LESS10_plot = np.array([ 0.06414970012741306, 0.0988036277611567, 0.06837743574826802, 0.013078492496022599, 0.05025960739905154])*100
MLP_result_LESS10_plot = np.array([ 0.0695552788946262, 0.0977371788182599, 0.06731797090667033, 0.0130784924960226, 0.0500640157803493])*100
RF_result_LESS10_plot = np.array([ 0.06710614935608303, 0.09690236717263745, 0.06684809209624684, 0.013343139638295056, 0.04964439080868164])*100
GP_result_LESS10_plot = np.array([0.06833071412535462,0.099,0.069,0.019,0.06])*100
#plot
x_ax = range(1, len(Ridge_result_LESS10_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_LESS10_plot, color="green", marker='o',markersize = 5, linewidth=0.7, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_LESS10_plot, color="blue", marker='v', markersize = 5, linewidth=0.7, label="MLP")
#RF
plt.plot(x_ax, RF_result_LESS10_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")
#GP
plt.plot(x_ax,GP_result_LESS10_plot,'mD-',markersize = 5, linewidth=0.7, label="GP")

plt.xticks(x_ax, ('b17', 'b18', 'b19', 'b20', 'b21'))

plt.ylabel('LESS10(%)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.show()

