# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt

Ridge_result_MAE_plot = [0.12386843565165045, 75.7612244794656, 50.04634842232905, 74.13077984331956, 374.1132485566584, 100.09363098148886]
MLP_result_MAE_plot = [1.8592719930257506, 75.21291909484509, 50.09456458359211, 74.16816887022449, 375.2785082288736, 99.94091309562408]
RF_result_MAE_plot = [0.20476553814296486, 86.91918068314574, 57.610878828294396, 85.33625341196115, 432.6371416123396, 112.52693003915876]

#plot
x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_MAE_plot, color="green", marker='o',markersize = 5, linewidth=0.7, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_MAE_plot, color="blue", marker='v', markersize = 5, linewidth=0.7, label="MLP")
#RF
plt.plot(x_ax, RF_result_MAE_plot, color="red", marker='s', markersize = 5, linewidth=0.7, label="RF")

plt.ylabel('MAE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(True)
plt.show()