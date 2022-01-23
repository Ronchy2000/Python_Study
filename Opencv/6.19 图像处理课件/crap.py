import cv2

img = cv2.imread('data/qrcode.jpg')     #读取图片
qrcode = cv2.QRCodeDetector()       #实例化
result, points, code = qrcode.detectAndDecode(img)      #检测与解码
