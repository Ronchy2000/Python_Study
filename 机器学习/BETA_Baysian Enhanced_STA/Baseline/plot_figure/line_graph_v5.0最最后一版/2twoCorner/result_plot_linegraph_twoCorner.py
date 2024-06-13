# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 14:22
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_plot_linegraph_twoCorner.py
# @Software: PyCharm

#result of two corners
import matplotlib.pyplot as plt
import numpy as np

##result of two corner
##MAE
Ridge_result_MAE_plot = [83.43214672325949, 59.44477926043379, 17.73982119379536, 145.75583980320224, 44.942620745463636, 36.88933352082417, 90.21917856160952,73.45374971757938, 73.01342612035783, 74.36636456037472]
MLP_result_MAE_plot = [57.06772679431526, 53.198229665971525, 17.00665503448923, 122.65913017503745, 38.048607856781736, 30.42478531689225, 79.07229049995004,63.95714113948849, 68.38874309579954, 68.36353255023145]
RF_result_MAE_plot = [68.06072351195603, 53.24573601554987, 17.276974260715786, 177.9374211581506, 48.205539393716286, 39.265615952709666, 94.46814641919934,62.50008188963776, 69.39005803098551, 70.03653661039459]
GP_result_MAE_plot = [26.10738182067871, 30.676011721293133, 15.985563112894695,26.801708857218426, 20.03329086303711, 14.917869567871094,46.4116096496582,45.00798797607422, 56.1288948059082, 57.16481908162435]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([11121.406432391499, 10726.316636388565, 642.6313759513055, 40981.7800980818, 4405.3336638217725, 2994.6455244655285, 15327.95581387677,10607.299657466707, 10183.187276244144, 10931.606255151803]) )
MLP_result_RMSE_plot =np.sqrt(np.array([5698.161702301267, 9186.974451734635, 560.160754471498, 29315.773455862185, 3106.2319519607895, 2094.2331535344633, 12235.572170748283,8001.3604152579865, 8725.259369659629, 9400.194200802347]))
RF_result_RMSE_plot = np.sqrt(np.array([7724.02395391465, 8916.365372767777, 549.1999669602999, 51476.92286025771, 4623.919956601782, 3151.782387341271, 15563.07618262879,7547.365269599999, 8837.604966629808, 9464.999169179051]))
GP_result_RMSE_plot = np.sqrt(np.array([2938.4017333984375, 4245.41796875, 692.3756103515625,2975.7457682291665, 2902.247517903646, 1505.702901204427,10342.621907552084,6732.477376302083, 7745.037923177083, 7640.37548828125]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.19822222222222222, 0.576, 0.8346666666666667, 0.17888888888888888, 0.5109259259259259, 0.5879629629629629, 0.2773662551440329,0.3517695473251029, 0.3274074074074074, 0.3234567901234568])*100
MLP_result_LESS30_plot = np.array([0.34844444444444445, 0.6515555555555556, 0.8568888888888889, 0.22796296296296295, 0.5585185185185185, 0.6507407407407407, 0.3190123456790123,0.3506172839506173, 0.33037037037037037, 0.3674074074074074])*100
RF_result_LESS30_plot = np.array([0.2659259259259259,  0.6666666666666666, 0.8777777777777778, 0.13657407407407407, 0.44320987654320987, 0.5439814814814815, 0.2074074074074074,0.39190672153635114, 0.3238683127572016, 0.35541838134430725])*100
GP_result_LESS30_plot = np.array([0.736, 0.7733333333333333, 0.8862222222222222, 0.7411111112, 0.8322222222222222, 0.8744444444444445, 0.6506995884773663,0.5853497942386832, 0.44098765432098763, 0.4434567901234568])*100



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
plt.ylim(10,180)#X轴范围

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
plt.ylim(5,95)#X轴范围

fig3_file = "line_graph_LESS30_plot.pdf"
plt.savefig(fig3_file,  bbox_inches='tight')
# plt.show()
