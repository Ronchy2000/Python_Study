# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

##Ridge_regression
Ridge_result_MAE_plot= np.array([[11.41786754,  7.40462112 , 8.24659632 , 1.50353389,  2.79771235],
 [ 0.82038444 , 1.77211006 , 1.94111045 , 2.69599613 , 6.58812984],
 [ 9.06245739, 12.38060682, 2.61342874,  1.22145088,  3.23182787],
 [ 9.24765074, 12.33217108,  3.8900278,   2.78359258 ,14.49320496]])
Ridge_result_RMSE_plot= np.sqrt(np.array([[293.37967128, 135.5279796,  196.98902984 , 18.53595472  ,52.15826584],
 [  4.52726567 , 26.02732814 , 19.33379895  ,24.3047757  ,104.99625895],
 [173.37027413 ,414.15936585 , 33.99806671  ,12.14842706 , 71.09828884],
 [180.07301469 ,406.94999701 , 68.12245961  ,38.28401429 ,487.99824267]]))
Ridge_result_LESS10_plot= np.array([[0.59814815, 0.77407407, 0.74722222, 0.98518519, 0.94814815],
 [0.98981481, 0.97037037, 0.9462963 , 0.93518519, 0.76018519],
 [0.65925926, 0.62222222, 0.94259259, 0.97222222, 0.96759259],
 [0.66296296, 0.60555556, 0.91388889, 0.93518519, 0.46481481]])*100

##MLP
MLP_result_MAE_plot= np.array([[10.92800129 , 7.89310153,  8.22776285,  1.73819577,  2.72616657],
 [ 0.92241148,  2.24969206, 1.86936171, 2.2004333 ,  4.2139948 ],
 [ 7.06127678, 11.56431563, 2.66905206, 1.5884166 ,  3.113846  ],
 [ 6.10207738, 11.63691608, 4.37101679, 2.8930761 , 12.94240306]])
MLP_result_RMSE_plot= np.sqrt(np.array([[265.50839024, 137.27397367, 189.80851797 , 25.56332971 , 34.30466647],
 [  4.55354454 , 23.89566821 , 17.57588578 , 20.55164964 , 70.99231901],
 [124.97148762 ,334.83956234 , 27.60680164 , 12.07422507 , 71.61663905],
 [ 79.45084498 ,343.96999316 , 66.53365581 , 30.87108448 ,362.85951507]]))
MLP_result_LESS10_plot= np.array([[0.63611111 ,0.75648148 ,0.74166667 ,0.9787037  ,0.93240741],
 [0.98981481 ,0.96944444,0.95      , 0.91574074, 0.88611111],
 [0.78518519 ,0.60925926,0.94444444, 0.9712963 , 0.96111111],
 [0.81388889 ,0.63796296,0.8962963 , 0.94444444, 0.52685185]])*100

##RF
RF_result_MAE_plot= np.array([[ 9.13366544 , 6.14558967,  6.93029194 , 1.41084577  ,3.25385789],
 [ 1.74490334 , 1.88403527,  1.99341334 , 2.00859406, 3.99860134],
 [ 4.34852912 , 7.95662599,  2.47311082 , 1.70355158, 2.64479457],
 [ 4.85106746 , 8.89948487,  3.65928698 , 2.58088041,10.01856582]])
RF_result_RMSE_plot= np.sqrt(np.array([[240.06256414 ,100.06494199 ,172.26124429,  12.38421637,  46.15293383],
 [ 17.84826775,  26.46643645 , 27.6307329 ,  18.12373293 , 66.70054697],
 [ 91.5503776 , 309.37368489 , 38.50306275,  17.80677949 , 42.65528407],
 [ 96.77241294, 266.36240772 , 62.01667521,  36.13785145 ,313.92201355]]))
RF_result_LESS10_plot= np.array([[0.69722222 ,0.77685185 ,0.76388889, 0.96759259 ,0.8787037 ],
 [0.98796296, 0.94722222 ,0.94907407, 0.96203704, 0.8212963 ],
 [0.82222222, 0.66018519 ,0.92777778, 0.95462963, 0.92777778],
 [0.82222222, 0.67685185 ,0.89814815, 0.92777778, 0.48796296]])*100

##GP
GP_result_MAE_plot= np.array([[8.435846 ,   5.48503  ,  5.251639  ,  0.27572706,  2.8391416 ],
 [ 1.3283985,   1.7054948,  1.2954327 ,  2.0136864  , 3.996737  ],
 [ 1.9071912,   3.8575594,  2.5044105,   0.69633716,  1.091078  ],
 [ 1.8200475,   3.7859674,  3.113445,    2.9052262  , 5.408911  ]])
GP_result_RMSE_plot= np.sqrt(np.array([[248.90251 ,  130.92714  , 102.349754  ,  2.3011193  ,50.145    ],
 [  5.715872,   16.088291 ,   8.714168 ,  17.377518 ,  61.1048   ],
 [ 29.301706,  116.43932  ,  28.63251  ,   3.2233698,  17.50025  ],
 [ 27.203115,  140.98793  ,  41.623444 ,  32.59038  , 131.29529  ]]))
GP_result_LESS10_plot= np.array([[0.91444444, 0.96555556, 0.96888889, 0.99888889 ,0.97333333],
 [1. ,        1.   ,      1.     ,    1.    ,     0.98666667],
 [0.99333333 ,0.97666667, 0.99  ,     1.  ,       0.99444444],
 [0.99666667, 0.97666667, 0.99555556 ,0.98444444, 0.97666667]])*100


###Settings*******************************************************************************
#设置x轴标签
name_list = ["Corner1", 'Corner2', 'Corner3', 'Corner4','Corner5']
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


#******************************************************************************************
plt.rcParams['figure.figsize'] = figsize
plt.rcParams['font.sans-serif'] = ['Arial']

# ###plot MAE
# for i in range(Ridge_result_MAE_plot.shape[0]):
#         fig = plt.figure(i+1,dpi =dpi)
#
#         #plot
#         x_ax = range(1, Ridge_result_MAE_plot.shape[1] + 1)
#         #Ridge
#         plt.plot(x_ax, Ridge_result_MAE_plot[i,:], color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
#         #MLP
#         plt.plot(x_ax, MLP_result_MAE_plot[i,:], color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#         #RF
#         plt.plot(x_ax, RF_result_MAE_plot[i,:], color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
#         #GP
#         plt.plot(x_ax,GP_result_MAE_plot[i,:], color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")
#
#         plt.ylabel('MAE(ps)', font)   # set ystick label
#         plt.xlabel('Designs', font)  # set xstck label
#         plt.title("Use {} iteration".format(i+1))
#         # plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
#         plt.xticks(x_ax, name_list, rotation=40)
#         plt.tick_params(labelsize=lablesize) #刻度字体大小10
#
#         plt.gcf().subplots_adjust(top=0.93,
#         bottom=0.2,
#         left=0.18,
#         right=0.95,
#         hspace=0.2,
#         wspace=0.2)
#         # plt.show()
#         # fig1_file = "line_graph_MAE_plot.pdf"
#         # plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！


# ###plot RMSE
# for i in range(MLP_result_RMSE_plot.shape[0]):
#         fig = plt.figure(i+1,dpi =dpi)
#
#         #plot
#         x_ax = range(1, Ridge_result_RMSE_plot.shape[1] + 1)
#         #Ridge
#         plt.plot(x_ax, Ridge_result_RMSE_plot[i,:], color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
#         #MLP
#         plt.plot(x_ax, MLP_result_RMSE_plot[i,:], color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
#         #RF
#         plt.plot(x_ax, RF_result_RMSE_plot[i,:], color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
#         #GP
#         plt.plot(x_ax,GP_result_RMSE_plot[i,:], color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")
#
#         plt.ylabel('RMSE(ps)', font)   # set ystick label
#         plt.xlabel('Designs', font)  # set xstck label
#         plt.title("Use {} iteration".format(i+1))
#         # plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
#         plt.xticks(x_ax, name_list, rotation=40)
#         plt.tick_params(labelsize=lablesize) #刻度字体大小10
#
#         plt.gcf().subplots_adjust(top=0.93,
#         bottom=0.2,
#         left=0.18,
#         right=0.95,
#         hspace=0.2,
#         wspace=0.2)
#         plt.show()
#         # fig1_file = "line_graph_MAE_plot.pdf"
#         # plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！
#



##plot_LESS30

#plot less30
for i in range(Ridge_result_LESS10_plot.shape[0]):
        fig = plt.figure(i+1,dpi =dpi)

        #plot
        x_ax = range(1, Ridge_result_LESS10_plot.shape[1] + 1)
        #Ridge
        plt.plot(x_ax, Ridge_result_LESS10_plot[i,:], color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
        #MLP
        plt.plot(x_ax, MLP_result_LESS10_plot[i,:], color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="MLP")
        #RF
        plt.plot(x_ax, RF_result_LESS10_plot[i,:], color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="RF")
        #GP
        plt.plot(x_ax,GP_result_LESS10_plot[i,:], color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Proposed")

        plt.ylabel('LESS30(%)', font)   # set ystick label
        plt.xlabel('Corner', font)  # set xstck label
        plt.title("Use {} iteration".format(i+1))
        # plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
        plt.xticks(x_ax, name_list, rotation=40)
        plt.tick_params(labelsize=lablesize) #刻度字体大小10

        plt.gcf().subplots_adjust(top=0.93,
        bottom=0.2,
        left=0.18,
        right=0.95,
        hspace=0.2,
        wspace=0.2)
        plt.show()
#         # fig1_file = "line_graph_MAE_plot.pdf"
#         # plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！
#




