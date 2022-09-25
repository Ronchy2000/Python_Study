# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 21:37
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : plot分布概率_双y轴.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
x_axis = [0,10,20,30,40,50]


####Ridge
#覆盖率
Ridge_ratio1 = [95.55421686746988, 97.3132530120482, 98.43373493975903, 99.32530120481927, 99.71084337349397, 99.87951807228916]
#覆盖率/资源消耗
Ridge_ratio2 = [115.6965008711015, 93.68862982293547, 80.27277835622324, 69.5511090297586, 62.332383160401854, 56.9030891173703]

####MLP
#覆盖率
MLP_ratio1 = [96.75903614457832, 97.91566265060241, 98.59036144578313, 99.2289156626506, 99.63855421686748, 99.89156626506023]
#覆盖率/资源消耗
MLP_ratio2 = [100.37537204423894, 84.16013904560842, 74.8471189628427, 66.46410007304084, 61.027273788372, 56.66829631587166]

####RF
#覆盖率
RF_ratio1 = [96.44578313253011, 97.85140562248996, 97.9871485943775, 99.3574297188755, 99.58835341365462, 99.58835341365462]
#覆盖率/资源消耗
RF_ratio2 = [102.69299907051787, 88.42285758227474, 68.61459536777245, 67.54643107746972, 62.990817572060834, 59.85939237640785]


####GP
#覆盖率
GP_ratio1 = [96.87469879518073, 98.32409638554217, 98.6289156626506, 99.10843373493977, 99.59349397590361, 99.81084337349397]
#覆盖率/资源消耗
GP_ratio2 = [107.64078403164899, 87.61099349474576, 75.97724598531678, 69.22742656017328, 64.05468880128811, 56.58034165057797]



###Settings*******************************************************************************
#设置x轴标签
name_list = ['a = 0','a = 10','a = 20','a = 30','a = 40','a = 50']
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

x_ax = range(1, len(Ridge_ratio1) + 1)


#plot
# fig = plt.figure(1,dpi =dpi)
# ax1 = fig.add_subplot(111)
#
#
# ax1.plot(x_ax, Ridge_ratio1, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax1.plot(x_ax, MLP_ratio1, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax1.plot(x_ax, RF_ratio1, color="blue", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax1.plot(x_ax, GP_ratio1, color="red", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax1.set_ylabel('Coverage(%)',font)
#
#
# ax2 = ax1.twinx()
# ax2.plot(x_ax, Ridge_ratio2, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax2.plot(x_ax, MLP_ratio2, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax2.plot(x_ax, RF_ratio2, color="blue", marker='^', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax2.plot(x_ax, GP_ratio2, color="red", marker='*', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
# ax2.set_ylabel('Coverage/Consumption(%)',font)
#
# # plt.title('Ridge')
# plt.xticks(x_ax,name_list,rotation=40)
# plt.tick_params(labelsize=lablesize) #刻度字体大小10
# plt.show()




fig = plt.figure(1,dpi =dpi)

plt.plot(x_ax, Ridge_ratio1, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, MLP_ratio1, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, RF_ratio1, color="blue", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, GP_ratio1, color="red", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")

plt.ylabel('Coverage(%)',font)   # set ystick label
# plt.xlabel('', font)  # set xstck label

plt.xticks(x_ax,name_list,rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)
plt.grid(linewidth = grid_linewidth)

plt.show()


fig = plt.figure(2,dpi =dpi)

plt.plot(x_ax, Ridge_ratio2, color="green", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, MLP_ratio2, color="magenta", marker='s', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, RF_ratio2, color="blue", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")
plt.plot(x_ax, GP_ratio2, color="red", marker='o', markersize = markersize, linestyle=linestyle, linewidth=linewidth, label="Ridge")

plt.ylabel('Coverage/Consumption(%)',font)   # set ystick label
# plt.xlabel('', font)  # set xstck label

plt.xticks(x_ax,name_list,rotation=40)
plt.tick_params(labelsize=lablesize) #刻度字体大小10
plt.gcf().subplots_adjust(top=0.93,
bottom=0.2,
left=0.18,
right=0.95,
hspace=0.2,
wspace=0.2)
plt.grid(linewidth = grid_linewidth)

plt.show()