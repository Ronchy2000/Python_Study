# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

##Ridge_regression
Ridge_result_MAE_plot= np.array([[ 9.06117485,  8.31810005, 10.81473387,  8.65738423,  0.13960467],
 [ 8.31626524,  5.84903651,  9.15953549, 9.67938036,  1.49318085],
 [ 1.48704138,  1.30068964,  6.29953446, 2.43314215,  1.89927954],
 [ 3.37342267,  2.90994821,  6.68118718, 4.12895294,  4.635231  ]])
Ridge_result_RMSE_plot= np.sqrt(np.array([[1.65672371e+02, 1.57900702e+02, 2.25955377e+02, 1.60955055e+02,
  4.73126323e-02],
 [1.31485147e+02, 6.47660677e+01,1.71139640e+02, 1.53460722e+02,
  2.30111231e+01],
 [1.28684155e+01, 1.58496937e+01, 1.02771365e+02, 2.35750317e+01,
  2.36274272e+01],
 [3.81141181e+01, 4.04439257e+01, 1.10766642e+02, 6.16557970e+01, 3.24799063e+01]]))
Ridge_result_LESS10_plot= np.array([[0.65555556, 0.72444444, 0.65555556, 0.68888889, 1.        ],
 [0.70888889, 0.81777778,0.68888889, 0.63333333,0.97111111],
 [0.95333333, 0.97333333,0.84222222, 0.95777778,0.95777778],
 [0.93777778, 0.93333333,0.84444444, 0.92222222,0.96666667]])*100

##MLP
MLP_result_MAE_plot= np.array([[8.66112554 ,7.86142621, 8.43256035 ,8.44030052 ,1.3290431 ],
 [5.80714004,6.04214127, 6.56180467, 8.6574739 , 2.14713166],
 [1.68552355,1.80578442, 4.97134194, 2.20874151, 2.11429842],
 [3.28145155,2.58080245, 6.404544  , 3.48955263, 3.21418343]])
MLP_result_RMSE_plot= np.sqrt(np.array( [[137.59282682 ,157.99464696, 174.1304956,  145.2064302 ,   3.04493323],
 [ 75.08458911, 69.70571904 , 81.77007181 ,120.46376285 ,22.0661521 ],
 [ 11.67989793, 15.68958863 , 73.94905644 , 15.04487027 ,31.77548448],
 [ 35.42710072, 23.46036165 ,100.26359706 , 40.47493449 ,25.95130094]]))
MLP_result_LESS10_plot= np.array([[0.70666667 ,0.75555556 ,0.75333333 ,0.70444444 ,1.        ],
 [0.84      , 0.80888889, 0.76444444, 0.67111111, 0.98      ],
 [0.96888889, 0.97555556, 0.86888889, 0.97555556, 0.95111111],
 [0.90666667, 0.94666667, 0.79777778, 0.95111111, 0.95111111]])*100

##RF
RF_result_MAE_plot= np.array([[7.54111948, 6.82370584, 7.08220352 ,6.15432021, 2.62356638],
 [5.04319385 ,4.47979128 ,5.74290242,7.46785102, 3.43361554],
 [2.4667247  ,2.57769115 ,4.96849312,2.56597355, 3.48511519],
 [3.52878134 ,3.62760244 ,4.47400366,3.7501337 , 3.05530747]])
RF_result_RMSE_plot= np.sqrt(np.array( [[114.97361919, 128.60969248, 137.67006393,  79.66029315 , 11.06859973],
 [ 59.33274645  ,39.01084418 ,73.03358621 , 98.39757241 , 30.01432613],
 [ 10.57215147  ,12.12223059 ,65.3853009  , 12.235454   , 25.5893652 ],
 [ 30.41159349  ,25.70161266 ,53.66227633 , 35.92855974 , 20.7458709 ]]))
RF_result_LESS10_plot= np.array([[0.76444444, 0.80666667, 0.80888889, 0.82444444 ,0.98444444],
 [0.84666667,0.91333333 ,0.85333333, 0.71555556, 0.95777778],
 [0.98666667,0.98666667 ,0.85111111, 0.98666667, 0.96444444],
 [0.90222222,0.93111111 ,0.88666667, 0.92888889, 0.97555556]])*100

##GP
GP_result_MAE_plot= np.array([[8.416153,   7.519403 ,  7.163858 ,  7.956636,  0.18260515],
 [4.077539  ,5.900098  ,4.2487154 , 7.0391736 , 1.746975  ],
 [1.542128  ,1.7973776 ,2.3494458 , 2.198086  , 2.1525764 ],
 [2.5468132 ,3.508634  ,3.0929937 , 3.7158997 , 3.5038307 ]])
GP_result_RMSE_plot= np.sqrt(np.array([[1.3800900e+02, 1.4431192e+02, 1.4512750e+02, 1.3577759e+02 ,8.4069565e-02],
 [4.8652386e+01,6.6402832e+01, 4.3148811e+01 ,8.5434837e+01, 2.4745159e+01],
 [1.4297483e+01,1.4871617e+01, 1.9994457e+01 ,1.6082981e+01, 3.7534290e+01],
 [2.5332563e+01,4.5795689e+01, 3.4124691e+01 ,4.4763054e+01, 2.5618200e+01]]))
GP_result_LESS10_plot= np.array([[0.96533333 ,0.92666667, 0.93133333, 0.95066667, 1.],
 [1., 0.96666667, 1.         ,0.89733333 ,0.984],
 [1., 1.        , 0.93733333 ,1.         ,0.984],
 [1., 1.        , 0.93733333 ,0.932      ,1.   ]])*100

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

figsize = (4,3)
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
        plt.ylim(60,105)

        plt.gcf().subplots_adjust(top=0.93,
        bottom=0.2,
        left=0.18,
        right=0.95,
        hspace=0.2,
        wspace=0.2)
        # plt.show()
        plt.grid(linewidth=0.5)
        if i == 3:
            plt.legend(loc="lower right", prop=legend_fontsize)
        fig1_file = str(i) + ".pdf"
        plt.savefig(fig1_file,  bbox_inches='tight') #tight,否则底部会被截断！







