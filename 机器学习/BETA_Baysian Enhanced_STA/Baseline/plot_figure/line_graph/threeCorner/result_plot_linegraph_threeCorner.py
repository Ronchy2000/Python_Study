# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 14:26
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_plot_linegraph_threeCorner.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

##result of three corner
##MAE
Ridge_result_MAE_plot = [69.86637469232801, 79.27736609071678, 16.660813394654603, 125.16266435826793, 40.43038534453514, 33.208690710465866, 80.39495199210769]
MLP_result_MAE_plot = [56.42469554476406, 61.91241558383296, 16.014632455396573, 76.04408546755025, 30.25288819249488, 26.082443922214793, 62.87659349711509]
RF_result_MAE_plot = [62.06283854431655, 63.3922500199854, 17.864698580562056, 75.24373954998092, 46.94684304695288, 39.80299308679386, 84.3032600948388]
GP_result_MAE_plot = [25.03220558166504, 30.96916437149048, 13.892928123474121, 26.862624168395996, 15.31080675125122, 12.13123046875, 37.534955978393555]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([7830.610194906172, 15752.234458756866, 566.399019922345, 28941.263180332826, 3618.71995204125, 2479.972465144967, 12320.879205823734]) )
MLP_result_RMSE_plot =np.sqrt(np.array([5797.1853262380455, 9782.950941728746, 501.415340365743, 11848.534963148282, 2143.073778985824, 1680.9572087108972, 8228.499428624289]))
RF_result_RMSE_plot = np.sqrt(np.array([6706.926502261549, 11677.464440424461, 650.669944411469, 14938.74195641561, 4366.284589219491, 3049.5216763662165, 12822.524269745367]))
GP_result_RMSE_plot = np.sqrt(np.array([3214.86865234375, 3653.075668334961, 680.8285827636719, 2275.4160766601562, 1211.0187377929688, 808.1890319824219, 6435.6198730468755]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.23333333333333334, 0.488, 0.8453333333333334, 0.165, 0.5658333333333333, 0.6205555555555555, 0.31530864197530867])*100
MLP_result_LESS30_plot = np.array([0.372, 0.5453333333333333, 0.8653333333333333, 0.3175, 0.6619444444444444, 0.7175, 0.39037037037037037])*100
RF_result_LESS30_plot = np.array([0.29555555555555557, 0.5333333333333333, 0.8555555555555555, 0.12361111111111112, 0.4525462962962963, 0.5263888888888889, 0.23189300411522634])*100
GP_result_LESS30_plot = np.array([0.7906666666666666, 0.8146666666666667, 0.8946666666666667, 0.5419444444444445, 0.8611111111111112, 0.9005555555555556,0.6871604938271605])*100




###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19']
#set marker，line.
markersize = 6
linestyle = '-.'
linewidth = 1


#******************************************************************************************

plt.figure(1)
plt.title("three dominant Corners")
#plot
x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_MAE_plot, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_MAE_plot, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#RF
plt.plot(x_ax, RF_result_MAE_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
#GP
plt.plot(x_ax,GP_result_MAE_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")

plt.ylabel('MAE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
plt.grid(0)
plt.xticks(x_ax, name_list)
plt.show()
# plt.draw()
fig1_file = "line_graph_MAE_plot.eps"
plt.savefig(fig1_file,  bbox_inches='tight')


##Figure RMSE
plt.figure(2)
plt.title("three dominant Corners")
#plot
# x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_RMSE_plot, color="green", marker='o',markersize = markersize,linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_RMSE_plot,color="magenta", marker='s', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="MLP")
#RF
plt.plot(x_ax, RF_result_RMSE_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="RF")
#GP
plt.plot(x_ax, GP_result_RMSE_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Proposed")


plt.ylabel('RMSE(ps)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="lower right")  #set legend location
# plt.grid(0)
plt.draw()
plt.xticks(x_ax, name_list)
plt.show()
fig2_file = "line_graph_RMSE_plot.eps"
plt.savefig(fig2_file,  bbox_inches='tight')




####Figure LESS30
plt.figure(3)
plt.title("three dominant Corners")
#plot
x_ax = range(1, len(Ridge_result_LESS30_plot) + 1)
#Ridge
plt.plot(x_ax, Ridge_result_LESS30_plot, color="green", marker='o',  markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_LESS30_plot, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#RF
plt.plot(x_ax, RF_result_LESS30_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="RF")
#GP
plt.plot(x_ax, GP_result_LESS30_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Proposed")
# plt.plot(x_ax,GP_result_LESS30_plot,'mD-',markersize = 6, linestyle='-.', linewidth=1, label="GP")


plt.ylabel('LESS30(%)')   # set ystick label
plt.xlabel('Designs')  # set xstck label
plt.legend(loc="upper right")  #set legend location
# plt.grid(0)
plt.xticks(x_ax, name_list)
plt.show()
plt.draw()
fig3_file = "line_graph_LESS30_plot.eps"
plt.savefig(fig3_file,  bbox_inches='tight')

