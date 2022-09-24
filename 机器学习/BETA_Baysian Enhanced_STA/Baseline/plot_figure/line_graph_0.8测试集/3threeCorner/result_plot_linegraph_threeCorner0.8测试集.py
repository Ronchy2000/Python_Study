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
Ridge_result_MAE_plot=[73.98367560849404, 81.09159476306064, 17.31390228433901, 125.94568477095984, 40.1280734088171, 33.25212299133574, 81.66485527228105,71.1981539870457, 71.29548561709096, 75.46202565205834]
MLP_result_MAE_plot = [62.62561942612176, 73.78975622633845, 23.789869528526964, 113.38654345579101, 34.88822957708717, 29.0024218398635, 69.85272789981894,61.29015401711483, 65.51082413970693, 71.1433737472114]
RF_result_MAE_plot = [54.776647288750965, 68.0793902815311, 16.384633863755692, 123.3381199477898, 38.875961310052546, 33.336061323555015, 72.91331424779632,67.79734891951964, 70.55913106494184, 75.04017546037106]
GP_result_MAE_plot =[45.912742614746094, 55.22666358947754, 16.023224353790283, 68.05660247802734, 31.89083766937256, 25.9942569732666, 53.16838073730469,54.81804656982422, 56.3601131439209, 62.40318489074707]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([8714.133027250733, 16168.083978275708, 599.269311992189, 29853.851825529156, 3564.6113663402475, 2565.0150611436065, 12699.45149700292,10176.147200379102, 9563.579069520592, 11804.634864804804]) )
MLP_result_RMSE_plot =np.sqrt(np.array([6563.162537872651, 13531.16588607496, 1546.0147231472745, 23252.759810106603, 2835.347233035716, 1872.8507278845075, 9821.465656246364,7335.908067086807, 7979.669950196216, 10445.165793505108]))
RF_result_RMSE_plot = np.sqrt(np.array([5697.402325019139, 11517.98442693483, 539.3065416214033, 28754.309006869047, 3255.0596504176415, 2293.6987234444805, 10804.77463762301,8628.653631783394, 8625.609067329817, 10881.674107500792]))
GP_result_RMSE_plot = np.sqrt(np.array( [5773.531005859375, 8653.814086914062, 556.4916076660156, 11265.4501953125, 2959.9547119140625, 2372.435546875, 7607.093017578125,6892.550048828125, 7162.458984375, 8931.260986328125]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.22916666666666666, 0.4816666666666667, 0.8391666666666666, 0.16640625, 0.5644965277777778, 0.6272569444444445, 0.3076388888888889,0.37546296296296294, 0.3351851851851852, 0.33271604938271604])*100
MLP_result_LESS30_plot = np.array([0.31625, 0.49916666666666665, 0.74125, 0.16927083333333334, 0.609375, 0.6670138888888889, 0.3528549382716049,0.39035493827160495, 0.3483024691358025, 0.35725308641975306])*100
RF_result_LESS30_plot = np.array( [0.3875, 0.5120833333333333, 0.86375, 0.21692708333333333, 0.5752604166666667, 0.6176215277777778, 0.32939814814814816,0.3427469135802469, 0.26851851851851855, 0.32083333333333336])*100
GP_result_LESS30_plot = np.array([0.5054166666666666, 0.5854166666666667, 0.8583333333333333, 0.4097222222222222, 0.6734375, 0.7348958333333333, 0.4960648148148148,0.45817901234567904, 0.4222993827160494, 0.42376543209876544])*100





###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19_v1', 'b19_v2', 'b19_v3', 'b19_v4']
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

fig3_file = "line_graph_LESS30_plot.pdf"
plt.savefig(fig3_file,  bbox_inches='tight')
# plt.show()

