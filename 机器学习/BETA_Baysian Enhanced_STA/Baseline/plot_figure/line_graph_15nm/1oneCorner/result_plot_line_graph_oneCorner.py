# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

## V5.0版本，
#  1.MAE，LESS30加了 grid，grid_linewidth控制线宽
## 2.固定y轴刻度
## 3.修改x-label名称

##MAE
Ridge_result_MAE_plot = [11.695651669738014, 14.07686871026499, 11.511793834456556,13.615445234791085, 14.955552095178774, 13.832483616951388,11.507725134034798, 13.156793328274452, 13.692133033616324, 17.35323448757774 ]
MLP_result_MAE_plot = [9.210556153790145, 12.325298546155596, 10.099835312696564, 12.429201359551914, 13.783526572229352, 13.534536943175514,10.434087339800321, 12.439886710872948, 13.19496136207132, 16.61567506369074 ]
RF_result_MAE_plot = [8.434790907783007, 10.59998330895436, 9.124731452197812,12.259880660530197, 15.41237409502595, 14.330175817131352, 12.640125785814046, 15.054135222530913, 15.184802857481476, 16.869173395608346]
GP_result_MAE_plot = [6.588776707649231, 9.369179487228394, 8.408998966217041,10.712541699409485, 11.789746761322021, 12.789071559906006, 10.016018629074097, 10.317225575447083, 11.576340913772583, 13.802766561508179]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([234.83976327986701, 388.0782429280629, 256.27244603080237,398.03989657003865, 513.3296978505059, 414.73811878223177,325.17432322698653, 435.9097855284259, 405.1292166157885, 700.4100230009242 ]) )
MLP_result_RMSE_plot =np.sqrt(np.array([153.3016518418116, 313.1626324050807, 208.13379492677717, 299.4922950779764, 419.91087076152536, 378.4398164129543,246.36979660154984, 365.13053681499264, 375.76128863363203, 612.5780907678908 ]))
RF_result_RMSE_plot = np.sqrt(np.array([139.25235525199952, 229.93486333827198, 178.97509677539816, 286.40267534882594, 527.5985253441854, 426.24036661147557, 293.45821110195124, 510.93209005567877, 413.7525130844443, 629.4524053205873]))
GP_result_RMSE_plot = np.sqrt(np.array([144.3258876800537, 316.5849380493164, 238.3371810913086, 293.3842468261719, 422.4942932128906, 410.8443908691406,278.38821029663086, 264.4979019165039, 278.0931739807129, 507.9977607727051 ]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.9471999999999999, 0.8829333333333335, 0.9314666666666668, 0.8909444444444444, 0.8630000000000001, 0.8693333333333333,0.9047710843373494, 0.8772771084337349, 0.878578313253012, 0.8182409638554216 ])*100
MLP_result_LESS30_plot = np.array([0.974, 0.9062666666666667, 0.9385333333333333,0.9241666666666667, 0.8720555555555556, 0.8708333333333332, 0.9286746987951807, 0.892409638554217, 0.8845060240963856, 0.8310361445783132])*100
RF_result_LESS30_plot = np.array([0.9672222222222222, 0.9383333333333334, 0.9494444444444444,0.9268518518518518, 0.850462962962963, 0.8664351851851851, 0.9237951807228916, 0.8766064257028112, 0.8802208835341365, 0.8459839357429719 ])*100
GP_result_LESS30_plot = np.array([0.9873333333333334, 0.964, 0.9486666666666667,0.9644444444444444, 0.935, 0.8905555555555555, 0.9226506024096386, 0.9350602409638554, 0.9186746987951807, 0.893132530120482])*100


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

legend_fontsize={ 'size': 7}

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
#
plt.legend(loc="upper right", prop=legend_fontsize)  #set legend location

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
plt.ylim(0,20)#y轴范围

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
plt.ylim(70,100)#y轴范围
fig3_file = "line_graph_LESS30_plot.pdf"
plt.savefig(fig3_file,  bbox_inches='tight')
# plt.show()




