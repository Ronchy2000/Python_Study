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
Ridge_result_MAE_plot = [10.77573818681161, 13.282575807907058, 9.52484833023033, 13.652467239135284, 14.762704596113654, 12.636491303061383,10.953169952143734, 13.758399816087154, 11.788230886403205, 17.66304772699776 ]
MLP_result_MAE_plot = [9.287893356570635, 9.807236718116316, 7.657170431785818,12.947131561872629, 13.853569612463781, 11.325477102637896, 8.727457945193077, 11.485061775541226, 10.409761130567963, 16.040615632921494]
RF_result_MAE_plot = [10.77573818681161, 13.282575807907058, 9.52484833023033, 12.984819889953101, 14.725773729422158, 12.860399892030198,11.589971507449405, 13.264462915115748, 13.728798807682075, 16.23198055950627 ]
GP_result_MAE_plot = [4.1406378746032715, 4.0825358629226685, 3.399733543395996, 3.866811752319336, 3.681896209716797, 3.5475778579711914, 4.24383819103241, 4.610795021057129, 3.944455623626709, 7.48351263999939]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([206.67777610499553, 282.51040899421395, 156.6552776163786,473.95245326334015, 560.8819853953464, 373.2342116796473, 269.0180895481657, 467.9416177856316, 299.25710676404293, 851.9600269534333 ]) )
MLP_result_RMSE_plot =np.sqrt(np.array([158.5002587083378, 158.29064338001652, 96.07049511831819,374.3367658295471, 454.8850380633205, 278.1089256686107, 167.4857439563083, 319.2442699164277, 227.805100227461, 707.8868083776403 ]))
RF_result_RMSE_plot = np.sqrt(np.array([206.67777610499553, 282.51040899421395, 156.6552776163786, 365.4300461376903, 487.0299764820194, 342.24642222717824, 279.0047580094445, 325.93520711121914, 325.7453510759395, 563.8457825908466]))
GP_result_RMSE_plot = np.sqrt(np.array([49.80710506439209, 47.067955017089844, 36.3958625793457,75.73760795593262, 66.93767166137695, 50.14269256591797, 75.11443901062012, 98.51364135742188, 69.83248329162598, 209.8824806213379 ]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.9506666666666667, 0.9373333333333334, 0.9786666666666667,0.9044444444444445, 0.8688888888888889, 0.9016666666666666,0.9257831325301205, 0.8575903614457832, 0.9004819277108433, 0.8332530120481928  ])*100
MLP_result_LESS30_plot = np.array([0.9733333333333334, 0.9746666666666667, 0.9973333333333333,0.9055555555555556, 0.8866666666666667, 0.9127777777777778, 0.9568674698795181, 0.9187951807228916, 0.9532530120481928, 0.844578313253012 ])*100
RF_result_LESS30_plot = np.array([0.9506666666666667, 0.9373333333333334, 0.9786666666666667,0.9305555555555556, 0.8689814814814815, 0.8949074074074074,  0.9239625167336011, 0.9119143239625167, 0.9215528781793842, 0.8508701472556894])*100
GP_result_LESS30_plot = np.array([0.9853333333333333, 0.992, 0.9933333333333333,0.9883333333333333, 0.9861111111111112, 0.9866666666666667,0.9872289156626506, 0.9768674698795181, 0.9833734939759036, 0.951566265060241 ])*100





###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17-v1', 'b17-v2', 'b17-v3', 'b18-v1','b18-v2','b18-v3', 'b19-v1', 'b19-v2', 'b19-v3', 'b19-v4']
#set marker，line.
markersize = 6
linestyle = '-.'
linewidth = 1
lablesize = 8 #设置坐标数字(字母)大小
font = {
        'weight' : 'bold',
        'size'   : 8}

legend_fontsize={ 'size': 10}

figsize = (4,3)
dpi = 150 #sci要求 300以上

grid_linewidth = 0.5 #网格线宽度



#******************************************************************************************
plt.rcParams['figure.figsize'] = figsize
plt.rcParams['font.sans-serif'] = ['Arial']


fig = plt.figure(1,dpi =dpi)

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

plt.ylabel('MAE(ps)', font)   # set ystick label
plt.xlabel('Designs', font)  # set xstck label

# plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10

plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)
# plt.show()

plt.grid(linewidth = grid_linewidth)
plt.ylim(2,20)#y轴范围

fig1_file = "line_graph_MAE_plot.pdf"
plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被阶段！






##Figure RMSE
plt.figure(2,dpi=dpi)
#Ridge
plt.plot(x_ax, Ridge_result_RMSE_plot, color="green", marker='o',markersize = markersize,linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
plt.plot(x_ax, MLP_result_RMSE_plot,color="magenta", marker='s', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="MLP")
#RF
plt.plot(x_ax, RF_result_RMSE_plot, color="blue", marker='^', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="RF")
#GP
plt.plot(x_ax, GP_result_RMSE_plot, color="red", marker='*', markersize = markersize, linestyle=linestyle,linewidth=linewidth, label="Proposed")

plt.ylabel('RMSE(ps)',font)   # set ystick label
plt.xlabel('Designs',font)  # set xstck label

# plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)

fig2_file = "line_graph_RMSE_plot.pdf"
plt.savefig(fig2_file,  bbox_inches='tight')
# plt.show()


####Figure LESS30
# plt.figure(3,figsize=figsize, dpi=dpi)
plt.figure(3,dpi=dpi)

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

plt.ylabel('LESS30(%)', font)   # set ystick label
plt.xlabel('Designs', font)  # set xstck label

# plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.xticks(x_ax, name_list, rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)

plt.grid(linewidth = grid_linewidth)
plt.ylim(80,100)#y轴范围

fig3_file = "line_graph_LESS30_plot.pdf"
plt.savefig(fig3_file,  bbox_inches='tight')
# plt.show()

