# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 10:55
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 计算概率估计并对比_Uncentainly.py
# @Software: PyCharm


import pandas as pd
import numpy as np

##使用累积分布函数
#Ridge
# df1 = pd.read_csv("../15nm实验/其他baseline/Ridge_15nm/b19_15nm_v5_prediction_Ridge.csv")
# df3 = pd.read_csv("../15nm实验/其他baseline/Ridge_15nm/b19_15nm_v5_real_Ridge.csv")


# #MLP
# df1 = pd.read_csv("../15nm实验/其他baseline/MLP_15nm/b19_15nm_v5_prediction_MLP.csv")
# df3 = pd.read_csv("../15nm实验/其他baseline/MLP_15nm/b19_15nm_v5_real_MLP.csv")

#
#
#
# #RF
# df1 = pd.read_csv("../15nm实验/其他baseline/RF_15nm/b19_15nm_v5_prediction_RF.csv")
# df3 = pd.read_csv("../15nm实验/其他baseline/RF_15nm/b19_15nm_v5_real_RF.csv")

#
#
#
# ##GP
df1 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15//b19_15nm_v5_prediction.csv")
df2 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15//b19_15nm_v5_covariance.csv")
df3 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15//b19_15nm_v5_real.csv")

##OK
# t0 = 1070
t0 = 220
df_data1 = np.array(df1.values[:, 1:]) - t0  #prediction
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:]) - t0  #real
pred_violation_ratio = np.sum( df_data1<0 )/(df_data1.shape[0]*df_data1.shape[1])
real_violation_ratio = np.sum( df_data3<0 )/(df_data3.shape[0]*df_data3.shape[1])

print("pred_violation_ratio:{}%".format(pred_violation_ratio*100))
print("real_violation_ratio:{}%".format(real_violation_ratio*100))
### violation占总路径的  xx%  左右
print("df_data1:",df_data1)
print("df_data3:",df_data3)


###第一种判定方法,
## slack < 0 ,判定为 violation
#用设置 余量 的方法  ，x-axis : 0  10  20  30  40  50
# delta = [0,10,15,20,25,30]
delta = [0,10,20,30,40,50]
#It is simple!

#指标：  1.准确率 （不画
#       2.资源消耗率
        #3.覆盖率
        #4. 覆盖率/资源消耗
plot_accuracy = []
resource_consumption_ratio = []

for dd in delta:
    df_data1 = np.array(df1.values[:, 1:]) - t0  # prediction
    df_data3 = np.array(df3.values[:, 1:]) - t0  # real
    #prediction
    df_data1[df_data1 < dd] = 0  #violation ->  0
    df_data1[df_data1 != 0] = 1
    #real
    df_data3[df_data3 < 0] = 0  #violation ->  0
    df_data3[df_data3 != 0] = 1
    # print("df_data1:",df_data1)
    # print("df_data3:",df_data3)
    diff_matrix = df_data1 - df_data3

    ##判断准确率 _ 不需要
    num_judge_error = np.sum(diff_matrix != 0)
    error_ratio = num_judge_error/(df_data1.shape[0]*df_data2.shape[1])
    print("判断准确率:{}%".format((1-error_ratio)*100))
    plot_accuracy.append((1-error_ratio)*100)

    ###判断覆盖率
    sum1 = np.sum(diff_matrix == 1)
    # print("差异数量：",sum1)
    ratio1 = (sum1 / (df_data1.shape[0] * df_data1.shape[1]))
    # print("预测覆盖率:", (1 - ratio1) * 100)
    plot_accuracy.append((1 - ratio1) * 100)

    ###判断资源消耗
    judge_violation = np.sum(df_data1 == 0)
    real_violation  = np.sum(df_data3 == 0)
    # print("预测违规:{}个,真实违规:{}个".format(judge_violation,real_violation))
    # print("资源消耗倍数:",judge_violation/real_violation)
    resource_consumption_ratio.append(judge_violation/real_violation)

#
#
#
#
# # print("resource_consumption_ratio=",resource_consumption_ratio)
# ## 覆盖率/资源消耗
# print("覆盖率:",plot_accuracy)
# print("覆盖率/资源消耗:",list(np.array(plot_accuracy)/np.array(resource_consumption_ratio)))
# # print("资源消耗/覆盖率:",np.array(resource_consumption_ratio/np.array(plot_accuracy)))
#


