# -*-coding:utf-8 -*-
 
import cv2 as cv
import numpy as np

src_image = cv.imread("./test1.jpg")

qrcoder = cv.QRCodeDetector()

codeinfo, points, straight_qrcode = qrcoder.detectAndDecode(src_image)

cv.drawContours(src_image, [np.int32(points)], 0, (0, 0, 255), 2)

print("qrcode :", codeinfo)
cv.imshow("result", src_image)
cv.waitKey(0)
