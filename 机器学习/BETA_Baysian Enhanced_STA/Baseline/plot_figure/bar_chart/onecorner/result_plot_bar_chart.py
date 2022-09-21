# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 23:55
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : result_plot_bar_chart.py
# @Software: PyCharm
##
'''
group bar chart.
这个图画出来，group内的每个element之间有间距

'''



import numpy as np
import matplotlib.pyplot as plt


##MAE
Ridge_result_MAE_plot = [111.30132809817754, 57.90079059731588, 17.273904424443664, 157.32547192624403, 66.36891834766786, 48.83213168636948, 125.82300093776868]
MLP_result_MAE_plot = [98.01578503469811, 55.049662424799976, 16.55337208986829, 140.08156517100534, 59.38977094958916, 52.51621600648719,114.52690422859027]
RF_result_MAE_plot = [76.31960083849353, 64.89200145220647, 16.969483480553198, 161.44262969615414, 63.77078862135144, 54.28026908761689, 116.34388125814507]
GP_result_MAE_plot = [69.16186714172363, 50.991258144378662, 16.70198760032654, 114.95974349975586, 38.12703323364258, 31.54359531402588, 71.5509033203125]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([20960.882955096586, 9982.765240790297, 638.8560937249046, 45542.85606003766, 9154.566199345067, 4739.935828936492, 27793.15565676238]) )
MLP_result_RMSE_plot =np.sqrt(np.array([18005.63240078954, 9218.374012075514, 557.7368960290847, 36385.6816289802, 7107.374069463624, 4981.291222343434, 23083.026416110777]))
RF_result_RMSE_plot = np.sqrt(np.array([10754.56969271128, 9766.493689749592, 607.504448450383, 47205.51471025095, 7381.942028740206, 5129.59672569583, 22406.457173726078]))
GP_result_RMSE_plot = np.sqrt(np.array([12933.08203125, 8276.563264465332, 568.4475341796875, 27347.8193359375, 3888.9388427734375, 2834.46630859375, 15377.767578125]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.1806666666666667, 0.5748, 0.8429333333333332, 0.14961111111111108, 0.3958055555555556, 0.47350000000000003, 0.19407407407407407])*100
MLP_result_LESS30_plot = np.array([0.23186666666666672, 0.6169333333333333, 0.8822666666666666, 0.18422222222222223, 0.4091111111111111, 0.38902777777777775, 0.19871604938271606])*100
RF_result_LESS30_plot = np.array([0.27666666666666667, 0.29, 0.8772222222222222, 0.19837962962962963, 0.30752314814814813, 0.3353009259259259, 0.15925925925925927])*100
GP_result_LESS30_plot = np.array([0.3592, 0.6354666666666667, 0.8810666666666667, 0.2608333333333333, 0.5881944444444445, 0.6548611111111111, 0.45444444444444443])*100

#plot

###Settings*******************************************************************************
#设置x轴标签
name_list = ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19']
#set
width = 0.07  #直方图的柱宽
# colors = ['b','g'] #直方图的颜色

barwidth = 0.15
#******************************************************************************************
x1 = np.array(range(len(name_list)))
x2 = [(i + barwidth) for i in x1]
x3 = [(i + barwidth) for i in x2]
x4 = [(i + barwidth) for i in x3]


#plot MAE
plt.figure(1)
plt.ylabel("MAE(ps)")  #纵轴标签
plt.xlabel("Designs")  #横轴标签

plt.bar(x1, Ridge_result_MAE_plot,color='#7f6d5f',  width=width,align='center', label='Ridge')
plt.bar(x2, MLP_result_MAE_plot ,color='#BB9B4B', width=width,align='center', label='MLP')
plt.bar(x3, RF_result_MAE_plot,color='#2d7f5e', width=width,align='center', label='RF')
plt.bar(x4, GP_result_MAE_plot,color='red', width=width,align='center', label='Proposed')

# Add xticks on the middle of the group bars
plt.xlabel('Design', fontweight='bold')
plt.xticks([r + 1.5*barwidth for r in range(len(x1))], ['b17_v1', 'b17_v2', 'b17_v3', 'b18_v1','b18_v2','b18_v3', 'b19'])

plt.legend()
# plt.title("Accuracy of Different Datasets without Revoking Requests",fontsize=14)
plt.show()















