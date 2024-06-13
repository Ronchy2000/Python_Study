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
Ridge_result_MAE_plot = [85.8974796061218, 61.413712396303765, 18.437381290343072, 144.82385053108925, 44.51828505821826, 37.00745201750058, 91.00816241174276,73.56670365521181, 74.26608587101195, 73.85644901228498]
MLP_result_MAE_plot = [72.45098684942208, 55.72368240075036, 21.109926244390127, 128.41343489286044, 40.00859706520359, 32.67651776861978, 79.2282854294054,66.13270255973707, 70.5849206169872, 70.16464281519916]
RF_result_MAE_plot = [60.24624258557768, 51.61066849979749, 17.118688350740562, 124.77913625931039, 42.692384640737, 33.01725274242073, 79.92455440294528,64.31163753592247, 69.7864436634502, 70.62483018888751]
GP_result_MAE_plot = [55.44738896687826, 50.69095834096273, 17.16524314880371, 76.3250249226888, 39.11604690551758, 30.170196533203125, 76.86322784423828,62.146139780680336, 66.8816769917806, 69.46152877807617]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([11539.868731761062, 11069.108602693208, 668.5844089368546, 40719.084683864945, 4390.284678008041, 3016.4271740002328, 15412.781306797822,10723.049299589518, 10489.754725351191, 11052.771048884248]) )
MLP_result_RMSE_plot =np.sqrt(np.array([8865.858081375422, 9381.943971543556, 1064.0022138332922, 32253.370294039178, 3406.7747852546495, 2457.040823048066, 12126.503674068488,8479.728782239144, 9117.94533538311, 9819.005565484937]))
RF_result_RMSE_plot = np.sqrt(np.array([7014.604015104069, 8309.273817699714, 605.0815239380885, 30329.022346961487, 3656.4551891454157, 2483.559267800583, 12009.858423669963,8095.289567648153, 9023.589228500272, 9633.869423974904]))
GP_result_RMSE_plot = np.sqrt(np.array([6217.236328125, 8360.875162760416, 600.0315450032552, 15587.33056640625, 3380.71826171875, 2454.9300537109375, 11708.776041666666,8205.479654947916, 8824.549153645834, 9574.479817708334]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.18583333333333332, 0.5763888888888888, 0.8213888888888888, 0.1814814814814815, 0.5112268518518519, 0.5763310185185185, 0.2701131687242798,0.35982510288065844, 0.323559670781893, 0.34192386831275723])*100
MLP_result_LESS30_plot = np.array([0.24083333333333334, 0.6216666666666667, 0.7766666666666666, 0.22887731481481483, 0.5348958333333333, 0.6291666666666667, 0.31224279835390945,0.36270576131687243, 0.31584362139917693, 0.34732510288065843])*100
RF_result_LESS30_plot = np.array([0.36138888888888887, 0.6247222222222222, 0.8694444444444445, 0.22175925925925927, 0.5185763888888889, 0.6452546296296297, 0.27314814814814814,0.383179012345679, 0.31748971193415637, 0.35581275720164607])*100
GP_result_LESS30_plot = np.array([0.4005555555555555, 0.6775, 0.8955555555555555, 0.3947337962962963, 0.5925810185185185, 0.6631712962962963, 0.33828189300411524,0.42351851851851855, 0.3518930041152263, 0.3749897119341564])*100



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
