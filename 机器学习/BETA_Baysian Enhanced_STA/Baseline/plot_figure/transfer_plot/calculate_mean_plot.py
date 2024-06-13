# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 17:48
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : calculate_mean_plot.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

###Ridge
Ridge_result_LESS10_plot1= np.array([[0.65555556, 0.72444444, 0.65555556, 0.68888889, 1.        ],
 [0.70888889, 0.81777778,0.68888889, 0.63333333,0.97111111],
 [0.95333333, 0.97333333,0.84222222, 0.95777778,0.95777778],
 [0.93777778, 0.93333333,0.84444444, 0.92222222,0.96666667]])*100

Ridge_result_LESS10_plot2= np.array([[0.59814815, 0.77407407, 0.74722222, 0.98518519, 0.94814815],
 [0.98981481, 0.97037037, 0.9462963 , 0.93518519, 0.76018519],
 [0.65925926, 0.62222222, 0.94259259, 0.97222222, 0.96759259],
 [0.66296296, 0.60555556, 0.91388889, 0.93518519, 0.66481481]])*100

Ridge_result_LESS10_plot3= np.array([[0.81204819, 0.75220884, 0.80763052, 0.75461847, 0.69477912],
 [0.71004016, 0.63614458, 0.67108434, 0.72409639 ,0.55702811],
 [0.69477912, 0.64056225, 0.65060241, 0.70281124 ,0.78835341],
 [0.40883534, 0.36305221, 0.37389558, 0.38835341 ,0.68232932]])*100

Ridge_mean = ((Ridge_result_LESS10_plot1 + Ridge_result_LESS10_plot2 + Ridge_result_LESS10_plot3)/3)

###MLP
MLP_result_LESS10_plot1 = np.array([[0.70666667 ,0.75555556 ,0.75333333 ,0.70444444 ,1.],
 [0.84      , 0.80888889, 0.76444444, 0.67111111, 0.98      ],
 [0.96888889, 0.97555556, 0.86888889, 0.97555556, 0.95111111],
 [0.90666667, 0.94666667, 0.79777778, 0.95111111, 0.95111111]])*100
MLP_result_LESS10_plot2 = np.array([[0.63611111 ,0.75648148 ,0.74166667 ,0.9787037  ,0.93240741],
 [0.98981481 ,0.96944444,0.95      , 0.91574074, 0.88611111],
 [0.78518519 ,0.60925926,0.94444444, 0.9712963 , 0.96111111],
 [0.81388889 ,0.63796296,0.8962963 , 0.94444444, 0.62685185]])*100
MLP_result_LESS10_plot3 = np.array([[0.79759036, 0.7124498,  0.79839357, 0.7184739,  0.54297189],
 [0.74698795, 0.60240964, 0.737751   ,0.77710843 ,0.62168675],
 [0.75943775, 0.56907631, 0.6815261  ,0.73253012 ,0.60040161],
 [0.49156627, 0.437751  , 0.44457831 ,0.42690763 ,0.66385542]])*100

MLP_mean = ((MLP_result_LESS10_plot1 + MLP_result_LESS10_plot2 + MLP_result_LESS10_plot3)/3)
###RF
RF_result_LESS10_plot1 = np.array([[0.76444444, 0.80666667, 0.80888889, 0.82444444 ,0.98444444],
 [0.84666667,0.91333333 ,0.85333333, 0.71555556, 0.95777778],
 [0.98666667,0.98666667 ,0.85111111, 0.98666667, 0.96444444],
 [0.90222222,0.93111111 ,0.88666667, 0.92888889, 0.97555556]])*100
RF_result_LESS10_plot2 = np.array([[0.69722222 ,0.77685185 ,0.76388889, 0.96759259 ,0.8787037 ],
 [0.98796296, 0.94722222 ,0.94907407, 0.96203704, 0.8212963 ],
 [0.82222222, 0.66018519 ,0.92777778, 0.95462963, 0.92777778],
 [0.82222222, 0.67685185 ,0.89814815, 0.92777778, 0.68796296]])*100
RF_result_LESS10_plot3 = np.array([[0.76947791, 0.74176707, 0.75502008 ,0.72329317, 0.45983936],
 [0.79477912,0.6        ,0.7497992  ,0.71365462 ,0.62610442],
 [0.71726908,0.61726908 ,0.67590361 ,0.6935743  ,0.64859438],
 [0.48433735,0.43574297 ,0.50803213 ,0.46947791 ,0.65301205]])*100

RF_mean = ((RF_result_LESS10_plot1 + RF_result_LESS10_plot2 + RF_result_LESS10_plot3)/3)
###GP
GP_result_LESS10_plot1 = np.array([[0.96533333 ,0.92666667, 0.93133333, 0.95066667, 1.],
 [1., 0.96666667, 1.         ,0.89733333 ,0.984],
 [1., 1.        , 0.93733333 ,1.         ,0.984],
 [1., 1.        , 0.93733333 ,0.932      ,1.   ]])*100

GP_result_LESS10_plot2 = np.array([[0.91444444, 0.96555556, 0.96888889, 0.99888889 ,0.97333333],
 [1. ,        1.   ,      1.     ,    1.    ,     0.98666667],
 [0.99333333 ,0.97666667, 0.99  ,     1.  ,       0.99444444],
 [0.99666667, 0.97666667, 0.99555556 ,0.98444444, 0.97666667]])*100

GP_result_LESS10_plot3 = np.array([[0.89445783 ,0.8713253,  0.89493976 ,0.83710843 ,0.82891566],
 [0.96289157 ,0.93638554 ,0.93590361, 0.91903614, 0.92433735],
 [0.96096386 ,0.9739759  ,0.96578313, 0.96385542, 0.90361446],
 [0.97771084 ,0.98975904 ,0.96216867, 0.97457831, 0.95759036]])*100

GP_mean = ((GP_result_LESS10_plot1 + GP_result_LESS10_plot2 + GP_result_LESS10_plot3)/3)
# print("GP_mean",GP_mean)
GP = np.mean(GP_mean,1)
print("GP",GP)


Ridge = Ridge_mean.mean(axis = 1)
MLP = MLP_mean.mean(axis = 1)
RF = RF_mean.mean(axis = 1)
GP = GP_mean.mean(axis = 1)


###Settings*******************************************************************************
#设置x轴标签
name_list = ["1", '2', '3', '4']
#set marker，line.
markersize = 6
linestyle = '-.'
linewidth = 1
lablesize = 8 #设置坐标数字(字母)大小
font = {
        'weight' : 'bold',
        'size'   : 8}
legend_fontsize={ 'size': 10}

figsize = (4.3,3.3)
dpi = 150 #sci要求 300以上

grid_linewidth = 0.5 #网格线宽度
#******************************************************************************************
plt.rcParams['figure.figsize'] = figsize
plt.rcParams['font.sans-serif'] = ['Arial']



plt.figure(1,dpi =dpi)
#plot
x_ax = range(1, 5)
#Ridge
plt.plot(x_ax, Ridge, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
#MLP
plt.plot(x_ax, MLP, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#RF
plt.plot(x_ax, RF, color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
#GP
plt.plot(x_ax,GP, color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")
plt.ylabel('LESS30(%)', font)   # set ystick label
plt.xlabel('Iteration', font)  # set xstck label
# plt.title("Use {} iteration".format(i+1))
# plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
plt.grid(linewidth=0.5)
plt.legend(loc="lower left", prop=legend_fontsize)
plt.xticks(x_ax, name_list)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.ylim(60,100)
plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)

plt.show()

fig1_file ="transfer_mean_lot.pdf"
plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！










