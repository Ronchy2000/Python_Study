# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 14:43
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : hello Plot2.py
# @Software: PyCharm
#鼠标作为画笔
# #学习以下功能：cv.setMouseCallback()

import cv2 as cv
import numpy as np
'''
我们创建一个鼠标回调函数，该函数在发生鼠标事件时执行。
鼠标事件可以是与鼠标相关的任何事物，例如左键按下，左键按下，左键双击等。
它为我们提供了每个鼠标事件的坐标(x，y)。通过此活动和地点，我们可以做任何我们喜欢的事情。
要列出所有可用的可用事件
'''
# events = [i for i in dir(cv ) if 'EVENT' in i]
# print( events)


#鼠标回调函数,双击画圆
# def draw_circle(event,x,y,flags,param):
#     if event ==cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img,(x,y),100,(255,0,0),-1)
#
# img = np.zeros((512,512,3),np.uint8)
#
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#     cv.imshow('image',img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
#
# cv.destroyAllWindows()



drawing = False # 如果按下鼠标，则为真
mode = True # 如果为真，绘制矩形。按 m 键可以切换到曲线
ix,iy = -1,-1
# 鼠标回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3),np.uint8)

cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()