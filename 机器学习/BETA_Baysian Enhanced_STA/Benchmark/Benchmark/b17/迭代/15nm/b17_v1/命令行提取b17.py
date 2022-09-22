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
'''
食用说明：
    本脚本为提取STA分析报告中的slack，start，end，生成针对某条路径的命令行指令。
    
    requirement:  正则表达式库
        conda/pip install re
tips：  
    1.请修改log文件路径；
    2.请修改所保存的txt文件名 
'''


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
            start = re.findall(".*Start-point  : (.*)/.*", line)
            Start_point.append(start)

        ret3 = re.match("End-point    :", line)
        if ret3:
            end = re.findall(".*End-point    : (.*)/.*", line)
            End_point.append(end)

    # print(len(timing_slack))
    # print(len(Start_point))
    # print(len(End_point))
    fileHandler.close()
    return timing_slack,Start_point,End_point






if __name__ =="__main__":
    file1_path = "b17_15_fast1.log"
    file2_path = "b18_15_fast1.log"
    file3_path = "b19_15_fast1.log"
    tmp_timing,tmp_start,tmp_end = extract_file_data(file1_path)

    with open("命令行_b17_15nm.txt","w+") as f:
        for i in range(len(tmp_start)):
            order = 'report timing -from  '+ str(tmp_start[i][0]) + ' -to  ' + str(tmp_end[i][0])+'\n'
            f.write(order)
        f.close()
    print("=========Done!=========")


