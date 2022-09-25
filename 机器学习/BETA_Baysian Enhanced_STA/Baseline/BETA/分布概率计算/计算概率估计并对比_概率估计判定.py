# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 22:00
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 计算概率估计并对比_概率估计判定.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from 分布概率 import prob_not_violation

# df1 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_prediction2.csv")
# df2 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_covariance2.csv")
# df3 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_real2.csv")
df1 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v5_prediction.csv")
df2 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v5_covariance.csv")
df3 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v5_real.csv")

#t0 = 1140  #b17_15nm_v1
# t0 = 1070 #b19_15nm_v1_real
t0 = 200
df_data1 = np.array(df1.values[:, 1:]) - t0  #prediction
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:]) - t0  #real
pred_violation_ratio = np.sum( df_data1<0 )/(df_data1.shape[0]*df_data1.shape[1])
real_violation_ratio = np.sum( df_data3<0 )/(df_data3.shape[0]*df_data3.shape[1])

print("pred_violation_ratio:{}%".format(pred_violation_ratio*100))
print("real_violation_ratio:{}%".format(real_violation_ratio*100))
# print("df_data1:",df_data1)
# print("df_data3:",df_data3)

plot_accuracy = []
resource_consumption_ratio = []

###第二种判定方法
##用概率估计判定，注意需要指定概率估计的阈值

# delta = [0,10,15,20,25,30]
delta = [0,10,20,30,40,50]
#指标：  1.准确率
#       2.资源消耗率

###指定概率估计的阈值
##prob > threshold为1，通过  ;   prob < threshold为0，violation
threshold = 0.5 #不能改
##predict
for dd in delta:
    df_data1 = np.array(df1.values[:, 1:]) - t0  # prediction
    df_data3 = np.array(df3.values[:, 1:]) - t0  # real

    predict_pass_prob = prob_not_violation(dd, df_data1, np.sqrt(df_data2))
    predict_pass_prob[predict_pass_prob > threshold] = 1  #通过
    predict_pass_prob[predict_pass_prob != 1] = 0  #不通过
    print("predict_pass_prob:",predict_pass_prob)
    ##real
    df_data3[df_data3 < 0] = 0  #slack < 0, violation ->  0
    df_data3[df_data3 != 0] = 1

    diff_matrix = predict_pass_prob - df_data3

    ###判断准确率 __不需要
    # num_judge_error = np.sum(diff_matrix != 0)
    # error_ratio = num_judge_error / (df_data1.shape[0] * df_data1.shape[1])
    # print("判断准确率:{}%".format((1 - error_ratio) * 100))
    # plot_accuracy.append((1 - error_ratio) * 100)

    ###判断覆盖率
    sum1 = np.sum(diff_matrix == 1)
    print("差异数量：", sum1)
    ratio1 = (sum1 / (df_data1.shape[0] * df_data1.shape[1]))
    print("预测覆盖率:", (1 - ratio1) * 100)
    plot_accuracy.append((1 - ratio1) * 100)

    ###判断资源消耗
    judge_violation = np.sum(predict_pass_prob == 0)
    real_violation  = np.sum(df_data3 == 0)
    print("预测违规:{}个,真实违规:{}个".format(judge_violation,real_violation))
    print("资源消耗倍数:",judge_violation/real_violation)
    resource_consumption_ratio.append(judge_violation/real_violation)


print("resource_consumption_ratio=",resource_consumption_ratio)
## 覆盖率/资源消耗
print("覆盖率:",plot_accuracy)
print("覆盖率/资源消耗:",np.array(plot_accuracy)/np.array(resource_consumption_ratio))
# print("资源消耗/覆盖率:",np.array(resource_consumption_ratio/np.array(plot_accuracy)))