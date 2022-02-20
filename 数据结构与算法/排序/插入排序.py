# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 插入排序.py
# @Time      : 2022/1/30 下午6:19
# @Author    : Ronchy
#插入排序

def insert_sort(li):
    for i in range( 1,len(li)-1 ): #i 摸到的牌的下标
        j = i -1 # 手里的牌的下标
        while li[j] > li[i] and j >=0 : #手上的牌比摸到的牌大

