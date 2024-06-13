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

##v3.0
# 在"[","]" 前添加反斜杠，作为转义字符

def str_insert(str_origin,pos,str_add):
    str_list = list(str_origin)
    cnt = 0
    str_out = ''
    for i in pos:
        str_list.insert(i+cnt,str_add)
        str_out = ''.join(str_list)
        cnt += 1
    return str_out

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
            timing_slack.append(slack)

        ret2 = re.match("Start-point  :",line)
        if ret2:
            tmp = re.findall(".*Start-point  : (.*)/.*", line)
            if not tmp : #列表为空
                start = re.findall(".*Start-point  : (.*)", line)[0]  #没有反斜杠的情况
            else:
                start = tmp[0]
            index_start = [index for index,i in enumerate(start) if i == "[" or i == "]" ]
            #不存在字符，返回-1
            if index_start != []:
                start = str_insert(start,index_start,'\\')
            Start_point.append(start)

        ret3 = re.match("End-point    :", line)
        if ret3:
            tmp = re.findall(".*End-point    : (.*)/.*", line)
            if not tmp : #列表为空
                end = re.findall(".*End-point    : (.*)", line)[0]  #没有反斜杠的情况
            else:
                end = tmp[0]
            index_end = [index for index, i in enumerate(end) if i == "[" or i == "]"]
            # 不存在字符，返回-1
            if index_end != 0:
                end = str_insert(end, index_end, '\\')
            End_point.append(end)

    # print(len(timing_slack))
    # print(len(Start_point))
    # print(len(End_point))
    fileHandler.close()
    return timing_slack,Start_point,End_point





if __name__ =="__main__":
    file1_path = "b17_15_fast1.log"
    file2_path = "b18_ff_VTL.log"
    file3_path = "b19_15_fast1.log"

    ###修改为需要提取的文件
    tmp_timing,tmp_start,tmp_end = extract_file_data( file2_path )

    print(tmp_start)
    print(tmp_end)
    print(len(tmp_end),len(tmp_end))

    start = np.array(tmp_start).reshape(-1,1)
    end = np.array(tmp_end).reshape(-1, 1)
    print(start.shape,end.shape)
    with open("命令行_b17_15nm.txt","w+") as f:
        for i in range(len(tmp_start)):
            order = 'report timing -from  '+ str(start[i][0]) + ' -to  ' + str(end[i][0])+'\n'
            f.write(order)
        f.close()
    print("=========Done!=========")


