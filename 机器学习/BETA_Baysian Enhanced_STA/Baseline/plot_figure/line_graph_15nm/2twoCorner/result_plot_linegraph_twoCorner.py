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
Ridge_result_MAE_plot = [10.611783403985676, 12.458972903530091, 10.574893570390016, 12.824955281487439, 13.391894212869024, 12.085297110390684, 10.6994320427335, 12.364392607073867, 11.761736998391134, 15.200017218978735]
MLP_result_MAE_plot = [9.3601039422964, 10.01126218493005, 9.017199332013178, 11.730734306074199, 12.583593113128915, 11.164943348491327, 9.912667993039504, 11.044571788616409, 10.743620465123788, 13.548437125872018]
RF_result_MAE_plot = [13.284755016175714, 14.841624288265768, 12.484526015516229,11.939669504764618, 13.338626470922796, 12.869753315255695, 11.589971507449405, 13.264462915115748, 13.728798807682075, 16.23198055950627 ]
GP_result_MAE_plot = [6.724053541819255, 4.569641192754109, 3.511625051498413,5.154624621073405, 6.11087433497111, 6.41828727722168, 5.379764874776204, 6.669436454772949, 4.783997535705566, 8.509540875752768]
##RMSE
Ridge_result_RMSE_plot = np.sqrt( np.array([194.659133688565, 281.79979940622337, 219.07623666229702, 404.94536426059676, 449.9408956671779, 327.2176369398019,285.8994117698004, 406.87044568246694, 291.90720778366807, 658.2709014662045 ]) )
MLP_result_RMSE_plot =np.sqrt(np.array([150.76715264254503, 197.16441848993517, 159.98530439711178,308.89843461866167, 364.8396402828994, 280.5334262733409, 244.77549180388465, 294.76206003845647, 235.8927387955963, 488.11975962030334 ]))
RF_result_RMSE_plot = np.sqrt(np.array([329.430853027713, 409.21388590245925, 297.7880906791184, 300.203626608094, 373.3035375264985, 334.55548353070077, 279.0047580094445, 325.93520711121914, 325.7453510759395, 563.8457825908466]))
GP_result_RMSE_plot = np.sqrt(np.array([95.4869016011556, 71.04464721679688, 42.89022890726725,109.25802357991536, 129.84888712565103, 119.34242502848308, 108.09951400756836, 147.36140950520834, 83.42117818196614, 306.4443715413411 ]))
#LESS30
Ridge_result_LESS30_plot = np.array([0.9573333333333334, 0.9333333333333333, 0.9493333333333334, 0.9148148148148149, 0.8944444444444445, 0.9107407407407407,0.9164658634538153, 0.8804819277108433, 0.917429718875502, 0.8586345381526105 ])*100
MLP_result_LESS30_plot = np.array([0.9777777777777777, 0.9457777777777778, 0.9564444444444444, 0.9192592592592592, 0.904074074074074, 0.914074074074074,0.9383132530120482, 0.916144578313253, 0.9344578313253012, 0.8780722891566265 ])*100
RF_result_LESS30_plot = np.array([0.8866666666666667, 0.882962962962963, 0.902962962962963, 0.9429012345679012, 0.8870370370370371, 0.9018518518518519, 0.9239625167336011, 0.9119143239625167, 0.9215528781793842, 0.8508701472556894])*100
GP_result_LESS30_plot = np.array([0.9884444444444445, 0.984, 0.992,0.98, 0.9644444444444444, 0.9618518518518518, 0.9747791164658635, 0.9614457831325302, 0.980562248995984, 0.9370281124497992 ])*100



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
plt.ylim(0,20)#y轴范围

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
plt.ylim(70,100)#X轴范围

fig3_file = "line_graph_LESS30_plot.pdf"
plt.savefig(fig3_file,  bbox_inches='tight')
# plt.show()
