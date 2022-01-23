import cv2

img = cv2.imread("./test3.jpg")

cv2.imshow('original image', img)

HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


darker_hsv = HSV_img.copy()
darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2]
darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)


cv2.imshow('HSV format image', HSV_img)
cv2.imshow('new', darker_img)

cv2.waitKey()