# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 10:41
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : helloVideo.py
# @Software: PyCharm
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webCam")
    exit()
# print(cap.isOpened()) #摄像头打开,返回True
while cap.isOpened():
    ret,frame = cap.read()#逐帧捕获
    if not ret:
        print("Can't receive frame,Exiting...")
        break
    #gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    cv.imshow('frame',gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()