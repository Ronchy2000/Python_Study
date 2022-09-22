# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 18:17
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : data_extract_b17_迭代1.py
# @Software: PyCharm
import os
import numpy as np
import re #正则表达式
import pandas as pd
import xlwt

# fileHandler = open(file_path,"r")

def extract_file_data(path):
    print(path)
    fileHandler = open(path, "r")

    timing_slack = []
    Start_point = []
    End_point = []
    while 1:
        line = fileHandler.readline()
        if not line:
            break
        # print(line.strip())
        #使用中严格遵照正则表达式使用，错一个空格都不行
        ret1 = re.match("Timing slack :",line)
        if ret1:
            slack = re.findall(".*Timing slack :(.*)ps.*",line)
            timing_slack.append(slack,)

        ret2 = re.match("Start-point  :",line)
        if ret2:
            tmp = re.findall(".*Start-point  : (.*)/.*", line)
            if not tmp : #列表为空
                start = re.findall(".*Start-point  : (.*)", line)  #没有反斜杠的情况
            else:
                start = tmp
            Start_point.append(start)

        ret3 = re.match("End-point    :", line)
        if ret3:
            tmp = re.findall(".*End-point    : (.*)/.*", line)
            if not tmp : #列表为空
                end = re.findall(".*End-point    : (.*)", line)  #没有反斜杠的情况
            else:
                end = tmp
            End_point.append(end)


    # print(len(timing_slack))
    # print(len(Start_point))
    # print(len(End_point))
    fileHandler.close()
    return timing_slack,Start_point,End_point



file1_path = ".\\Benchmark\\b17\\迭代\\b17_迭代3\\b17_tt_VTL3.log"
file2_path = ".\\Benchmark\\b17\\迭代\\b17_迭代3\\b17_ff_VTL3.log"
file3_path = ".\\Benchmark\\b17\\迭代\\b17_迭代3\\b17_fs_VTL3.log"
file4_path = ".\\Benchmark\\b17\\迭代\\b17_迭代3\\b17_sf_VTL3.log"
file5_path = ".\\Benchmark\\b17\\迭代\\b17_迭代3\\b17_ss_VTL3.log"
# file1_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_ff_VTG3.log"


#tt
#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file1_path)
np_tmp_timing = np.array(tmp_timing)
np_tmp_start = np.array(tmp_start)
np_tmp_end = np.array(tmp_end)
# print(np_tmp_timing.shape)
# print(np_tmp_start.shape)
# print(np_tmp_end.shape)]

##用了numpy 不用列表生成器去reshape(-1,1)
# tmp_timing_col_tmp = [[r[col] for r in tmp_timing] for col in range(len(tmp_timing[0]))]
# tmp_start_col_tmp = [[r[col] for r in tmp_start] for col in range(len(tmp_start[0]))]
# tmp_end_col_tmp = [[r[col] for r in tmp_end] for col in range(len(tmp_end[0]))]


#ff
# #=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file2_path)

np_tmp_timing = np.concatenate((np_tmp_timing,tmp_timing), axis=1)
np_tmp_start = np.concatenate((np_tmp_start,tmp_start), axis=1)
np_tmp_end = np.concatenate((np_tmp_end,tmp_end), axis=1)

# print(np_tmp_timing,np_tmp_start,np_tmp_end)
print(np_tmp_timing.shape,np_tmp_start.shape,np_tmp_end.shape)

#fs
# #=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file3_path)

np_tmp_timing = np.concatenate((np_tmp_timing,tmp_timing), axis=1)
np_tmp_start = np.concatenate((np_tmp_start,tmp_start), axis=1)
np_tmp_end = np.concatenate((np_tmp_end,tmp_end), axis=1)

# print(np_tmp_timing,np_tmp_start,np_tmp_end)
print(np_tmp_timing.shape,np_tmp_start.shape,np_tmp_end.shape)

#sf
# #=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file4_path)

np_tmp_timing = np.concatenate((np_tmp_timing,tmp_timing), axis=1)
np_tmp_start = np.concatenate((np_tmp_start,tmp_start), axis=1)
np_tmp_end = np.concatenate((np_tmp_end,tmp_end), axis=1)

# print(np_tmp_timing,np_tmp_start,np_tmp_end)
print(np_tmp_timing.shape,np_tmp_start.shape,np_tmp_end.shape)

#ss
# #=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file5_path)

np_tmp_timing = np.concatenate((np_tmp_timing,tmp_timing), axis=1)
np_tmp_start = np.concatenate((np_tmp_start,tmp_start), axis=1)
np_tmp_end = np.concatenate((np_tmp_end,tmp_end), axis=1)

# print(np_tmp_timing,np_tmp_start,np_tmp_end)
print(np_tmp_timing.shape,np_tmp_start.shape,np_tmp_end.shape)
#
# #对timing进行处理,字符串-> number
# print("np_tmp_timing:\n",np_tmp_timing)
# for i in np_tmp_timing:
#     for j in i:
#
# print(int_timing)

# 判断start 与 end 是否一致
#这个数据比对方法很巧妙，https://blog.csdn.net/u011699626/article/details/107912910
def all_equal(lst):
  return lst[1:] == lst[:-1]

# print(all_equal([1, 2, 3, 4, 5, 6])) # False
# print(all_equal([1, 1, 1, 1])) # True

start_right = []
for cnt,row in enumerate(np_tmp_start):
    if all( all_equal(row) ) == True:
        start_right.append(cnt)
#统计下标
print("len(start_right)",len(start_right))

end_right = []
for cnt,row in enumerate(np_tmp_end):
    if all( all_equal(row) ) == True:
        end_right.append(cnt)
#统计下标
print("len(end_right)",len(end_right))

if len(start_right) == len(end_right):
    header = [('Corner' + str(i)) for i in range(1,np_tmp_timing.shape[1]+1)]
    df = pd.DataFrame(np_tmp_timing, columns=header)

    df.to_excel("b17_VTLx5.xls", sheet_name='Sheet1', index=True)

else:
    print("error，有end不一致")
    #用cnt删掉不一致的路径
    #留坑待填...



