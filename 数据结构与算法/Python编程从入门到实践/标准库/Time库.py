# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 21:42
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Time库.py
# @Software: PyCharm
import time,sys

t= time.gmtime()
#time.struct_time(tm_year=2021,tm_mon=8,tm_mday=25,tm_hour=1,tm_min=53,tm_sec=43,tm_wday=5,tm_yday=237,tm_isdst=0)
print(t.tm_zone)
# while True:
#     print(time.time()) #1970年1月1日到现在，过去的秒数
print(time.ctime())

#时间格式化 strftime(tpl,t)
t= time.gmtime()
print(time.strftime("%Y-%m-%d %H:%M:%S",t))

