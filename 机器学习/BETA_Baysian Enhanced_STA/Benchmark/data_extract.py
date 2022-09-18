# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 18:17
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : data_extract.py
# @Software: PyCharm
import os
import numpy as np
import re #正则表达式


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
        ret1 = re.match("Timing slack :",line)
        if ret1:
            slack = re.findall(".*Timing slack :(.*)ps.*",line)
            timing_slack.append(slack)

        ret2 = re.match("Start-point  :",line)
        if ret2:
            start = re.findall(".*Start-point  : (.*)", line)
            Start_point.append(start)

        ret3 = re.match("End-point    :", line)
        if ret3:
            end = re.findall(".*End-point    : (.*)", line)
            End_point.append(end)

    # print(len(timing_slack))
    # print(len(Start_point))
    # print(len(End_point))
    fileHandler.close()
    return timing_slack,Start_point,End_point


file1_path = "E:\\Developer\\Python\\Myworkshop\\Python_Study\\机器学习\\BETA_Baysian Enhanced_STA\\Benchmark\\Benchmark\\b17\迭代\\b17_迭代3\\b17_tt_VTG3.log"
file2_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_ff_VTG3.log"
file3_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_fs_VTG3.log"
file4_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_sf_VTG3.log"
file5_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_ss_VTG3.log"
# file1_path = "E:\Developer\Python\Myworkshop\Python_Study\机器学习\BETA_Baysian Enhanced_STA\Benchmark\Benchmark\\b17\迭代\\b17_迭代3\\b17_ff_VTG3.log"

timing,start,end  = [],[],[]

#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data("E:\\Developer\\Python\\Myworkshop\\Python_Study\\机器学习\\BETA_Baysian Enhanced_STA\\Benchmark\\Benchmark\\b17\迭代\\b17_迭代3\\b17_tt_VTG3.log")
# print(len(timing),len(start),len(end))
timing.append(tmp_timing,axis=1)
start.append(tmp_start,axis =1)
end.append(tmp_end,axis =1)
#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file2_path)
# print(len(timing),len(start),len(end))
timing.append(tmp_timing,axis=1)
start.append(tmp_start,axis =1)
end.append(tmp_end,axis =1)
#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file3_path)
# print(len(timing),len(start),len(end))
timing.append(tmp_timing,axis=1)
start.append(tmp_start,axis =1)
end.append(tmp_end,axis =1)
#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file4_path)
# print(len(timing),len(start),len(end))
timing.append(tmp_timing,axis=1)
start.append(tmp_start,axis =1)
end.append(tmp_end,axis =1)
#=============================================================
tmp_timing,tmp_start,tmp_end = extract_file_data(file5_path)
# print(len(timing),len(start),len(end))
timing.append(tmp_timing,axis=0)
start.append(tmp_start,axis =0)
end.append(tmp_end,axis =0)

print(len(timing),len(timing))














