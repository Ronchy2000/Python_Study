# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 10:55
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 计算概率估计并对比.py
# @Software: PyCharm


import pandas as pd
import numpy as np
from 分布概率 import prob_not_violation
##使用累积分布函数

df1 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_prediction2.csv")
df2 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_covariance2.csv")
df3 = pd.read_csv("../15nm实验/BETA_twoCorner/b17_15/b17_15nm_v1_real2.csv")

##OK
df_data1 = np.array(df1.values[:, 1:])
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:])

###计算相对误差
# relative_error = (np.abs(df_data1 - df_data3))/df_data3
# print("two dominant corners:",relative_error)


predict_pass_prob = prob_not_violation(1200, df_data1, np.sqrt(df_data2))
# print( predict_pass_prob )
threshold = 0.7
#广播，替换
predict_pass_prob[predict_pass_prob > threshold] = 1  #通过
predict_pass_prob[predict_pass_prob != 1] = 0  #不通过
# print("predict_pass_prob:",predict_pass_prob)


##real_data
clock_period = 1200.0
#注意逻辑！
df_data3[df_data3 > clock_period] = 0
df_data3[df_data3 != 0] = 1
# print("处理后:",df_data3)

#根据概率估计，判断与真实情况的差异
##这里用    矩阵减法  （不用循环
diff = predict_pass_prob - df_data3
# print("diff_matrix:",diff)

num_diff = np.sum(diff != 0)
print("概率估计错误的数量:",num_diff)
ratio = num_diff/(df_data1.shape[0]*df_data1.shape[1])
print("错误的数量占总预测的百分比: {}%".format(ratio*100))


sum1 = np.sum(diff > 0)
ratio1 = (sum1/(df_data1.shape[0]*df_data1.shape[1]))
# print("判断通过但实际不通过的数量{} ,百分比{}%".format(sum1,ratio1))


sum2 = np.sum(diff < 0)
ratio2 = (sum2/(df_data1.shape[0]*df_data1.shape[1]))
# print("判断不通过但实际通过的数量{} ,百分比{}%".format(sum2,ratio2))


print("预测覆盖率:",(1 - ratio1)*100)
print("消耗资源倍数:",1/ratio)
print("预测正确率:",(1-ratio)*100)




