# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

##Ridge_regression
Ridge_result_MAE_plot= np.array( [[ 9.35093288,  8.66903313,  7.50383171 , 8.51122045 ,12.87108185],
 [ 9.15123057, 11.31534218, 11.25895502 , 7.83444932 ,12.97698353],
 [ 8.72935584, 10.95052165,  9.47550641 , 8.09205542 ,26.93481853],
 [18.02540949, 18.59745977, 18.81273083 ,15.84696036 ,20.90104617]])
Ridge_result_RMSE_plot= np.sqrt(np.array([[ 289.48846547  ,255.025785 ,   183.69210887  ,229.89491664,  457.44521807],
 [ 238.56872537, 340.22201999 , 322.52590268 , 157.52819036,  370.6985823 ],
 [ 185.14337729, 278.48034063 , 211.19321652 , 136.91275136, 1374.84245554],
 [ 588.81544475, 592.65303907 , 660.34696628 , 399.31956892,  785.36408305]]))
Ridge_result_LESS10_plot= np.array([[0.81204819, 0.75220884, 0.80763052, 0.75461847, 0.69477912],
 [0.71004016, 0.63614458, 0.67108434, 0.72409639 ,0.55702811],
 [0.69477912, 0.64056225, 0.65060241, 0.70281124 ,0.78835341],
 [0.40883534, 0.36305221, 0.37389558, 0.38835341 ,0.68232932]])*100

##MLP
MLP_result_MAE_plot= np.array([[ 7.27835646 , 9.07009141,  7.78577932 , 8.55697697, 13.78466806],
 [ 8.34108853 ,10.98852237 , 9.94999051 , 7.7134053 , 12.33239583],
 [ 7.86851152 ,10.67997462 , 8.96799299 , 7.95905334, 20.06095651],
 [14.47285357 ,15.41608399 ,16.88157821 ,14.49678267, 20.70360972]])
MLP_result_RMSE_plot= np.sqrt(np.array([[181.26218009 ,244.07835674, 175.62802566, 215.62232908 ,408.970373  ],
 [176.55519503 ,258.56108426 ,245.91691104 ,138.51777943 ,356.86017379],
 [164.22928871 ,214.84841912 ,183.13602308 ,112.20691739 ,841.18935857],
 [416.42177454 ,411.61714633 ,576.05182867 ,361.4112969  ,778.27696442]]))
MLP_result_LESS10_plot= np.array([[0.79759036, 0.7124498,  0.79839357, 0.7184739,  0.54297189],
 [0.74698795, 0.60240964, 0.737751   ,0.77710843 ,0.62168675],
 [0.75943775, 0.56907631, 0.6815261  ,0.73253012 ,0.60040161],
 [0.49156627, 0.437751  , 0.44457831 ,0.42690763 ,0.66385542]])*100

##RF
RF_result_MAE_plot= np.array([[ 8.25017048 , 9.5235591  , 8.5194394  , 9.62110075, 13.81260759],
 [ 7.57294461 ,11.06038481,  8.56625736 , 8.60119676 ,10.98336977],
 [ 8.60987032 ,10.33315246,  9.37727201 , 7.67719922 ,20.11049094],
 [14.53237619 ,15.15768878, 14.29654863 ,12.54798634 ,18.12954756]])
RF_result_RMSE_plot= np.sqrt(np.array([[170.04508183, 223.85887552, 177.82456791, 219.18004277, 367.73728984],
 [147.65610727, 221.3722451  ,206.18461294, 149.55167192, 262.69592747],
 [175.32965944, 220.544954   ,183.05213328, 108.21783122, 765.91452283],
 [419.8151234 , 415.92170187 ,389.23709962, 259.9334758 , 541.92578716]]))
RF_result_LESS10_plot= np.array([[0.76947791, 0.74176707, 0.75502008 ,0.72329317, 0.45983936],
 [0.79477912,0.6        ,0.7497992  ,0.71365462 ,0.62610442],
 [0.71726908,0.61726908 ,0.67590361 ,0.6935743  ,0.64859438],
 [0.48433735,0.43574297 ,0.50803213 ,0.46947791 ,0.65301205]])*100

##GP
GP_result_MAE_plot= np.array([[ 7.5807652  ,8.289959 ,  7.317842  , 8.27231   ,13.270262 ],
 [ 4.5718536 , 5.627351  ,5.273333  , 6.2281675 , 6.607679 ],
 [ 3.7830856 , 3.599036  ,3.5872078 , 4.3812103 , 7.179072 ],
 [ 5.803155  , 5.1851034 ,4.9892144 , 5.797836  , 6.4171114]])
GP_result_RMSE_plot= np.sqrt(np.array( [[166.87698, 237.08894 , 168.36572,  218.53476 , 374.97015 ],
 [ 83.25604  ,116.61146 ,117.86236 , 104.14024 , 125.98526 ],
 [ 64.02202  , 56.07797 , 58.38645 ,  50.25168 , 271.37    ],
 [159.38     ,152.44235 ,151.75594 , 101.088165, 222.88875 ]]))
GP_result_LESS10_plot= np.array([[0.89445783 ,0.8713253,  0.89493976 ,0.83710843 ,0.82891566],
 [0.96289157 ,0.93638554 ,0.93590361, 0.91903614, 0.92433735],
 [0.96096386 ,0.9739759  ,0.96578313, 0.96385542, 0.90361446],
 [0.97771084 ,0.98975904 ,0.96216867, 0.97457831, 0.95759036]])*100


###Settings*******************************************************************************
#设置x轴标签
name_list = ["1", '2', '3', '4','5']
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
        plt.xlabel('Corner ID', font)  # set xstck label
        # plt.title("Use {} iteration".format(i+1))
        # plt.legend(loc="upper left", prop=legend_fontsize)  #set legend location
        plt.xticks(x_ax, name_list)
        plt.tick_params(labelsize=lablesize) #刻度字体大小10
        plt.ylim(35,105)

        plt.gcf().subplots_adjust(top=0.93,
        bottom=0.2,
        left=0.18,
        right=0.95,
        hspace=0.2,
        wspace=0.2)
        # plt.show()
        plt.grid(linewidth=0.5)
        if i == 0:
            plt.legend(loc="lower left", prop=legend_fontsize)
        fig1_file = str(i) + ".pdf"
        plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！










