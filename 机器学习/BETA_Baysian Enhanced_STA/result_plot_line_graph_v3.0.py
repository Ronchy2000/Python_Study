# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


##MAE
Ridge_result_MAE_plot = [111.30132809817754, 57.90079059731588, 17.273904424443664, 157.32547192624403, 66.36891834766786, 48.83213168636948, 125.82300093776868]
MLP_result_MAE_plot = [98.01578503469811, 55.049662424799976, 16.55337208986829, 140.08156517100534, 59.38977094958916, 52.51621600648719,114.52690422859027]
RF_result_MAE_plot = [76.31960083849353, 64.89200145220647, 16.969483480553198, 161.44262969615414, 63.77078862135144, 54.28026908761689, 116.34388125814507]
GP_result_MAE_plot = [69.16186714172363, 25.991258144378662, 16.70198760032654, 123.95974349975586, 59.12703323364258, 46.54359531402588, 102.5509033203125]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([20960.882955096586, 9982.765240790297, 638.8560937249046, 45542.85606003766, 9154.566199345067, 4739.935828936492, 27793.15565676238]) )
MLP_result_RMSE_plot =np.sqrt(np.array([18005.63240078954, 9218.374012075514, 557.7368960290847, 36385.6816289802, 7107.374069463624, 4981.291222343434, 23083.026416110777]))
RF_result_RMSE_plot = np.sqrt(np.array([10754.56969271128, 9766.493689749592, 607.504448450383, 47205.51471025095, 7381.942028740206, 5129.59672569583, 22406.457173726078]))
GP_result_RMSE_plot = np.sqrt(np.array([10373.929809570312, 1651.8921508789062, 557.6262229919433, 30977.794921875, 7112.946520996094, 4549.913818359375, 20189.493408203125]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.1806666666666667, 0.5748, 0.8429333333333332, 0.14961111111111108, 0.3958055555555556, 0.47350000000000003, 0.19407407407407407])*100
MLP_result_LESS30_plot = np.array([0.23186666666666672, 0.6169333333333333, 0.8822666666666666, 0.18422222222222223, 0.4091111111111111, 0.38902777777777775, 0.19871604938271606])*100
RF_result_LESS30_plot = np.array([0.27666666666666667, 0.29, 0.8772222222222222, 0.19837962962962963, 0.30752314814814813, 0.3353009259259259, 0.15925925925925927])*100
GP_result_LESS30_plot = np.array([0.43546666666666667, 0.7793333333333333, 0.8765333333333334, 0.22855902777777778, 0.41519097222222223, 0.4490277777777778, 0.19493827160493826])*100

###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19']
#set marker，line.
markersize = 6
linestyle = '-.'
linewidth = 1

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


#******************************************************************************************

plt.figure(1)
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
# plt.grid(0)
plt.xticks(x_ax, name_list)
# plt.show()
# plt.draw()
fig1_file = "line_graph_MAE_plot.eps"
# plt.rcParams.update('font', **font)
# plt.rcParams.update({'font.size': 22})

plt.xticks(rotation=40)
plt.rcParams.update({'font.size': 16})
plt.savefig(fig1_file,  bbox_inches='tight')


##Figure RMSE
plt.figure(2)
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
# plt.draw()
plt.xticks(x_ax, name_list)
# plt.show()
fig2_file = "line_graph_RMSE_plot.eps"
# plt.savefig(fig2_file,  bbox_inches='tight')

plt.xticks(rotation=40)
plt.rcParams.update({'font.size': 16})
plt.savefig(fig2_file,  bbox_inches='tight')





####Figure LESS30
plt.figure(3)
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


# plt.show()
# plt.draw()
fig3_file = "line_graph_LESS30_plot.eps"

plt.xticks(rotation=40)
plt.rcParams.update({'font.size': 16})
plt.savefig(fig3_file,  bbox_inches='tight')



