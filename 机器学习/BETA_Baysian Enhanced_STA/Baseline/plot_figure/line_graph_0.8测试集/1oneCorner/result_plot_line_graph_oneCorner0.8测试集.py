# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 17:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_MAE_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


##MAE
Ridge_result_MAE_plot = [113.23581435131977, 58.29545267439005, 17.739185926382895, 156.35493570272953, 65.85628689888412, 49.038524180001914, 126.33447119777347,95.12141921338376, 86.71979315355756, 83.14405546122332]
MLP_result_MAE_plot = [106.85974904918896, 56.52332201864864, 17.616728483618974, 144.93789338848126, 59.596017495384636, 51.57404155542927, 110.53341287581009,90.59573350531078, 84.54381991613249, 81.73709573529511]
RF_result_MAE_plot = [76.10618574873789, 65.16136595966873, 16.886312235395344, 162.1235518504095, 64.20331635634209, 53.2701485562999, 116.87210649278086,86.08636114359582, 80.10085201832989, 78.06773253973282]
GP_result_MAE_plot = [68.09783267974854, 46.653852462768555, 18.75293493270874, 126.19697952270508, 48.98042583465576, 39.1057071685791, 97.92949104309082,82.40385246276855, 75.74958229064941, 72.99100494384766]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array( [21522.484345836507, 10090.460900468086, 678.0156218979819, 45244.36922832682, 9057.313037763908, 4817.762087310682, 27742.289302962443,16087.670575484499, 13896.887145697914, 12562.60998550921]) )
MLP_result_RMSE_plot =np.sqrt(np.array([19878.651134156156, 9640.363212137203, 904.3659453667657, 38278.2018726562, 7170.591825869315, 4793.958523651771, 21095.921921209763,14627.402725230424, 12966.337769334343, 12165.86849639721]))
RF_result_RMSE_plot = np.sqrt(np.array( [11233.005155797464, 10331.59100910085, 637.6231025173154, 48262.11380949866, 7538.801163610557, 4907.747019202539, 22441.78037991296,13613.304968922597, 12028.234290488854, 11197.537595917438]))
GP_result_RMSE_plot = np.sqrt(np.array([9384.675537109375, 7388.4278564453125, 611.74755859375, 33398.3583984375, 5060.6524658203125, 3383.9617309570312, 18831.607177734375,13151.683349609375, 12862.991455078125, 12700.91552734375]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.17591666666666667, 0.5818749999999999, 0.8432499999999999, 0.15032118055555554, 0.39523437499999997, 0.4728819444444444, 0.18950617283950616,0.21412037037037038, 0.23958333333333334, 0.2119212962962963])*100
MLP_result_LESS30_plot = np.array([0.18766666666666668, 0.6008333333333333, 0.8597083333333334, 0.16977430555555556, 0.40763888888888894, 0.41302083333333334, 0.19212962962962962,0.21871141975308642, 0.2458719135802469, 0.25200617283950616])*100
RF_result_LESS30_plot = np.array([0.28208333333333335, 0.28729166666666667, 0.8866666666666667, 0.1916232638888889, 0.29266493055555554, 0.35520833333333335, 0.16057098765432098,0.2607253086419753, 0.29552469135802467, 0.2804012345679012])*100
GP_result_LESS30_plot = np.array([0.3491666666666667, 0.6860416666666667, 0.8435416666666666, 0.24661458333333333, 0.470703125, 0.5476996527777778, 0.2705246913580247,0.28912037037037036, 0.2570601851851852, 0.2683641975308642])*100

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




