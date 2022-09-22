# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


##MAE
Ridge_result_MAE_plot = [111.30132809817754, 57.90079059731588, 17.273904424443664, 157.32547192624403, 66.36891834766786, 48.83213168636948, 125.82300093776868,94.11091024168182, 86.07512294325065, 82.16493055969667]
MLP_result_MAE_plot = [98.01578503469811, 55.049662424799976, 16.55337208986829, 140.08156517100534, 59.38977094958916, 52.51621600648719,114.52690422859027,90.0395749560655, 83.47680704406268, 81.25271511972682]
RF_result_MAE_plot = [76.31960083849353, 64.89200145220647, 16.969483480553198, 161.44262969615414, 63.77078862135144, 54.28026908761689, 116.34388125814507,84.97938322589154, 89.45738800267449, 87.23042316495231]
GP_result_MAE_plot = [69.16186714172363, 50.991258144378662, 16.70198760032654, 114.95974349975586, 38.12703323364258, 31.54359531402588, 71.5509033203125,81.5212345123291, 83.95445442199707, 82.59679794311523]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([20960.882955096586, 9982.765240790297, 638.8560937249046, 45542.85606003766, 9154.566199345067, 4739.935828936492, 27793.15565676238,15600.112249173508, 13360.309213076422, 11994.420669279007]) )
MLP_result_RMSE_plot =np.sqrt(np.array([18005.63240078954, 9218.374012075514, 557.7368960290847, 36385.6816289802, 7107.374069463624, 4981.291222343434, 23083.026416110777,14627.530538165045, 12548.383231337459, 12217.746840133907]))
RF_result_RMSE_plot = np.sqrt(np.array([10754.56969271128, 9766.493689749592, 607.504448450383, 47205.51471025095, 7381.942028740206, 5129.59672569583, 22406.457173726078,12997.910874401174, 11860.83635018058, 10788.464249669132]))
GP_result_RMSE_plot = np.sqrt(np.array([12933.08203125, 8276.563264465332, 568.4475341796875, 27347.8193359375, 3888.9388427734375, 2834.46630859375, 15377.767578125,13271.770263671875, 12291.131103515625, 12512.67041015625]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.1806666666666667, 0.5748, 0.8429333333333332, 0.14961111111111108, 0.3958055555555556, 0.47350000000000003, 0.19407407407407407,0.21345679012345678, 0.2288888888888889, 0.21037037037037037])*100
MLP_result_LESS30_plot = np.array([0.23186666666666672, 0.6169333333333333, 0.8822666666666666, 0.18422222222222223, 0.4091111111111111, 0.38902777777777775, 0.19871604938271606,0.2317283950617284, 0.2502469135802469, 0.2762962962962963])*100
RF_result_LESS30_plot = np.array([0.27666666666666667, 0.29, 0.8772222222222222, 0.19837962962962963, 0.30752314814814813, 0.3353009259259259, 0.15925925925925927,0.2696502057613169, 0.2453909465020576, 0.26012345679012347])*100
GP_result_LESS30_plot = np.array([0.3592, 0.6354666666666667, 0.8810666666666667, 0.2608333333333333, 0.5881944444444445, 0.6548611111111111, 0.45444444444444443,0.3108641975308642, 0.26123456790123456, 0.27])*100

###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19_v1', 'b19_v2', 'b19_v3', 'b19_v4']
#set marker，line.
markersize = 6
linestyle = '-.'
linewidth = 1
lablesize = 10 #设置坐标数字(字母)大小
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 12}

legend_fontsize={ 'size': 10}

figsize = (6,4.5)
dpi = 300 #sci要求 300以上

plt.rcParams['figure.figsize'] = figsize
#******************************************************************************************

# plt.figure(1,figsize=figsize, dpi=dpi)
fig = plt.figure(1)
axes = fig.add_axes([0.2,0.2,0.7,0.7])

#plot
x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
axes.plot(x_ax, Ridge_result_MAE_plot, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
axes.plot(x_ax, MLP_result_MAE_plot, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#RF
axes.plot(x_ax, RF_result_MAE_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
#GP
axes.plot(x_ax,GP_result_MAE_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")

plt.ylabel('MAE(ps)', font)   # set ystick label
plt.xlabel('Designs', font)  # set xstck label

plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.rcParams['figure.figsize'] = (6.0, 4.5)
fig1_file = "./line_graph_MAE_plot.eps"
# plt.savefig(fig1_file,  bbox_inches='tight')
plt.savefig(fig1_file)
plt.show()

##Figure RMSE
# plt.figure(2,figsize=figsize, dpi=dpi)
fig = plt.figure(2)
axes = fig.add_axes([0.2,0.2,0.7,0.7])
#plot
# x_ax = range(1, len(Ridge_result_MAE_plot) + 1)
#Ridge
axes.plot(x_ax, Ridge_result_RMSE_plot, color="green", marker='o',markersize = markersize,linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
axes.plot(x_ax, MLP_result_RMSE_plot,color="magenta", marker='s', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="MLP")
#RF
axes.plot(x_ax, RF_result_RMSE_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="RF")
#GP
axes.plot(x_ax, GP_result_RMSE_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Proposed")


plt.ylabel('RMSE(ps)',font)   # set ystick label
plt.xlabel('Designs',font)  # set xstck label

plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.rcParams['figure.figsize'] = (6.0, 4.5)
fig2_file = "line_graph_RMSE_plot.eps"
# plt.savefig(fig2_file,  bbox_inches='tight')
plt.savefig(fig2_file)
plt.show()


####Figure LESS30
# plt.figure(3,figsize=figsize, dpi=dpi)
fig = plt.figure(3)
axes = fig.add_axes([0.2,0.2,0.7,0.7])
#plot
x_ax = range(1, len(Ridge_result_LESS30_plot) + 1)
#Ridge
axes.plot(x_ax, Ridge_result_LESS30_plot, color="green", marker='o',  markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Ridge")
#MLP
axes.plot(x_ax, MLP_result_LESS30_plot, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#RF
axes.plot(x_ax, RF_result_LESS30_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="RF")
#GP
axes.plot(x_ax, GP_result_LESS30_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Proposed")
# plt.plot(x_ax,GP_result_LESS30_plot,'mD-',markersize = 6, linestyle='-.', linewidth=1, label="GP")


plt.ylabel('LESS30(%)', font)   # set ystick label
plt.xlabel('Designs', font)  # set xstck label

# plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.rcParams['figure.figsize'] = (6.0, 4.5)
fig3_file = "line_graph_LESS30_plot.eps"
plt.savefig(fig3_file,  bbox_inches='tight')
plt.show()




