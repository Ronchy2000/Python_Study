# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 17:03
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Series.py
# @Software: PyCharm

import pandas as pd
a = [1,7,2,10]

index1 = [1,2,3,4]  #必须和a的长度匹配
myvar = pd.Series(a,index=index1)
print(myvar)
#------------------------------------------------------
#dict -> Series
calories = {"day1":420,"day2":380,"day3":390}
myvar1 = pd.Series(calories)
print(myvar1)




