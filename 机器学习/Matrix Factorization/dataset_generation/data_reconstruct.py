# -*- coding: utf-8 -*-
# @Time    : 2022/8/13 13:51
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : data_reconstruct.py
# @Software: PyCharm
import numpy as np
import pandas as pd

# df = pd.read_csv("timing1500x18.csv")
df = pd.read_csv("timing1500x14.csv")

data = np.array(df.values[:,1:])
print(data.shape)
output = np.zeros([data.shape[0]*data.shape[1] ,3],dtype=None)
# output[:2] = output[:2].astype(np.float16)

print("output.shape:",output.shape)
index = 0
for j in range(data.shape[1]):
    for i in range(data.shape[0]):
        #从0开始
        output[0+index, 0] = int(i)
        output[0+index, 1] = int(j)
        #从1开始
        # output[0+index, 0] = int(i)+1
        # output[0+index, 1] = int(j)+1

        output[0+index, 2] = data[i,j]
        index = index+1
print(output[-1,-1])
print(output)
df_out = pd.DataFrame(output,columns=["row","col","value"])
df_out.to_csv("timing1500x14_flattern.csv")