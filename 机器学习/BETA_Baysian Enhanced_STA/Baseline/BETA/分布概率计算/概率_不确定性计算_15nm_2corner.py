# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 4:42
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 概率_不确定性计算_template.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from 分布概率 import prob_not_violation

def uncentainty(a):
    #对应元素相乘
    ##nan 代表无穷小 ，inf代表无穷大
    # uncent = np.multiply((-a),np.log2(0.0000000001+a)) - np.multiply((1-a),np.log2(1-a+0.000000001))
    # uncent = np.multiply((-a),np.log2(a+1e-19)) - np.multiply((1-a),np.log2(1-a+1e-19))
    uncent = np.multiply((-a),np.log2(a)) - np.multiply((1-a),np.log2(1-a))
    return uncent

# df1 = pd.read_csv("../15nm实验/BETA_twoCorner/b19_15/b19_15nm_v5_prediction2.csv")
# df2 = pd.read_csv("../15nm实验/BETA_twoCorner/b19_15/b19_15nm_v5_covariance2.csv")
# df3 = pd.read_csv("../15nm实验/BETA_twoCorner/b19_15/b19_15nm_v5_real2.csv")
#
# df1 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v4_prediction.csv")
# df2 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v4_covariance.csv")
# df3 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v4_real.csv")
df1 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v2_prediction.csv")
df2 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v2_covariance.csv")
df3 = pd.read_csv("../15nm实验/BETA_oneCorner/b19_15/b19_15nm_v2_real.csv")

t0 = 830
df_data1 = np.array(df1.values[:, 1:]) - t0  #prediction
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:]) - t0  #real
pred_violation_ratio = np.sum( df_data1<0 )/(df_data1.shape[0]*df_data1.shape[1])
real_violation_ratio = np.sum( df_data3<0 )/(df_data3.shape[0]*df_data3.shape[1])
print("pred_violation_ratio:{}%".format(pred_violation_ratio*100))
print("real_violation_ratio:{}%".format(real_violation_ratio*100))


#===================================================================================

#指标：  1.覆盖率
#       2.资源消耗率

###指定概率估计的阈值
##prob > threshold为1，通过  ;   prob < threshold为0，violation
threshold = 0.5 #不能改

prob_matrix = prob_not_violation(0, df_data1, np.sqrt(df_data2))

##计算不确定性矩阵 值域 T
uncentain_matrix = uncentainty(prob_matrix)
print("uncentain_matrix:",uncentain_matrix.shape[0]*uncentain_matrix.shape[1])
print("nan的个数",uncentain_matrix.shape[0]*uncentain_matrix.shape[1] - np.sum(uncentain_matrix > 0))

#T0 0  ~   1 之间
# T0 = 1e-15  #min 曲线下方无任何点
#nan的个数不用check
# index = np.argwhere(uncentain_matrix > T0)  #此时该recheck
# print(index.shape[0])


#predict
prob_matrix[prob_matrix > threshold] = 1  #通过
prob_matrix[prob_matrix != 1] = 0  #不通过

cnt_judged_2check = np.sum(prob_matrix == 0)
print("cnt_judged_2check=",cnt_judged_2check)
print("cnt_judged_2check = ",np.sum(cnt_judged_2check != 0))

##real
df_data3[df_data3 < 0] = 0  #slack < 0, violation ->  0
df_data3[df_data3 != 0] = 1

cnt_real_needed_2check = np.sum(df_data3 == 0)
print("cnt_real_needed_2check=",cnt_real_needed_2check)
print("other_aspect = ",np.sum(df_data3 != 0))


diff_matrix = prob_matrix - df_data3
# print(np.sum(diff_matrix != 0))
# print(diff_matrix[diff_matrix !=0 ])



#### 判断错误的阈值
error_index = np.where(diff_matrix != 0)
list_error_index = np.dstack((error_index[0], error_index[1])).squeeze()
# for ii in error_index:
#     print(diff_matrix[ii[0],ii[1]])

T0_min = 1
print("与True对比后，错误的数量:",list_error_index.shape[0])
print("初始正确率:",(1 - list_error_index.shape[0]/(df_data3.shape[0]*df_data3.shape[1]))*100 )
for ii in list_error_index:
    tmp = uncentain_matrix[ii[0],ii[1]]
    if T0_min > tmp:
        T0_min = tmp
print("覆盖率100%的阈值",T0_min)
#error覆盖率100%的阈值 0.006175937669239268


















#####计算T0增大，error_coverage 减小
# variable_T0 = [0.0062,0.007,0.008,0.009,0.010]
# variable_T0 = [0.006,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.90,0.99,0.999]
variable_T0 = np.arange(0.006,0.9999,0.02)

cnt = 0
error_coverage_list = []
resource_consumption_list = []
extra_cost_list = []


for v_T0 in variable_T0:
    # bqd_index = np.argwhere(uncentain_matrix > v_T0)
    bqd_index = np.where(uncentain_matrix > v_T0)
    list_bqd_index = np.dstack((bqd_index[0], bqd_index[1])).squeeze()
    # print("list_bqd_index.shape[0]",list_bqd_index.shape[0])
    # print("list_error_index.shape[0]:",list_error_index.shape[0])
    ###重要，变成list 才能对比！！！
    for ii in range(list_error_index.shape[0]):
        for jj in range(list_bqd_index.shape[0]):
            if list_error_index[ii][0] ==list_bqd_index[jj][0] and list_error_index[ii][1] ==list_bqd_index[jj][1]:
            # print()
                cnt += 1
    error_coverage = cnt/list_error_index.shape[0]  #错误覆盖率
    error_coverage_list.append(error_coverage)

    resource_consumption = (cnt_judged_2check + cnt) / cnt_real_needed_2check  # 资源消耗率
    resource_consumption_list.append(resource_consumption)

    extra_cost = (cnt_judged_2check + cnt) / (df_data1.shape[0] * (5 - df_data1.shape[1]))  # extra cost
    extra_cost_list.append(extra_cost)

    cnt = 0

print("error_coverage_list =",error_coverage_list)
print("resource_consumption_list =",resource_consumption_list)
print("extra_cost_list =",extra_cost_list)



#
# #########计算T0增大，error_coverage 减小
# variable_T0 = np.arange(0.006,0.9999,0.02)
# cnt = 0
# error_coverage_list = []
# for v_T0 in variable_T0:
#     # bqd_index = np.argwhere(uncentain_matrix > v_T0)  # 坑死人
#     bqd_index = np.where(uncentain_matrix > v_T0) # 用这个
#     list_bqd_index = np.dstack((bqd_index[0], bqd_index[1])).squeeze()
#     # print("list_bqd_index.shape[0]",list_bqd_index.shape[0])
#     # print("list_error_index.shape[0]:",list_error_index.shape[0])
#     ###重要，变成list 才能对比！！！
#     for ii in range(list_error_index.shape[0]):
#         for jj in range(list_bqd_index.shape[0]):
#             if list_error_index[ii][0] ==list_bqd_index[jj][0] and list_error_index[ii][1] ==list_bqd_index[jj][1]:
#             # print()
#                 cnt += 1
#
#     resource_consumption = (cnt_judged_2check + cnt) / cnt_real_needed_2check #资源消耗率
#     resource_consumption_list.append(resource_consumption)
#     # print("cnt=",cnt)
#     #print("error_coverage:", resource_consumption)
#     cnt = 0
#
# print("resource_consumption_list:",resource_consumption_list)
#
#
#
#
#
# #########计算T0增大，extra_cost 减小
# variable_T0 = np.arange(0.006,0.9999,0.02)
# cnt = 0
# extra_cost_list = []
# for v_T0 in variable_T0:
#     # bqd_index = np.argwhere(uncentain_matrix > v_T0)  #坑死人
#     bqd_index = np.where(uncentain_matrix > v_T0) #用这个
#     list_bqd_index = np.dstack((bqd_index[0], bqd_index[1])).squeeze()
#     # print("list_bqd_index.shape[0]",list_bqd_index.shape[0])
#     # print("list_error_index.shape[0]:",list_error_index.shape[0])
#     ###重要，变成list 才能对比！！！
#     for ii in range(list_error_index.shape[0]):
#         for jj in range(list_bqd_index.shape[0]):
#             if list_error_index[ii][0] ==list_bqd_index[jj][0] and list_error_index[ii][1] ==list_bqd_index[jj][1]:
#             # print()
#                 cnt += 1
#
#     extra_cost = (cnt_judged_2check + cnt) / (df_data1.shape[0]*(5- df_data1.shape[1])) #extra cost
#     extra_cost_list.append(extra_cost)
#     cnt = 0
#
# print("extra_cost_list:",extra_cost_list)
#




