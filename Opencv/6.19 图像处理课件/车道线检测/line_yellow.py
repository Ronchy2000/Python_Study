import cv2
import numpy as np
import math

def calc_angle(x1, y1, x2, y2):
    b=math.atan2(y2-y1,x2-x1)
    c = 180 + b / math.pi * 180
    if c > 180:
        c = c - 180
    return c

#f = open("./line_yellow.txt", "w")

x1_points = [];x2_points = [];y1_points = [];y2_points = []
x1_fre = [];x2_fre = [];y1_fre = [];y2_fre = []
def stat_lines(x1,y1,x2,y2):   #计算点位
    if x1 not in x1_points:
        x1_points.append(x1)
        x1_fre.append(1)
    else:
        x1_fre[x1_points.index(x1)] += 1

    if x2 not in x2_points:
        x2_points.append(x2)
        x2_fre.append(1)
    else:
        x2_fre[x2_points.index(x2)] += 1

    if y1 not in y1_points:
        y1_points.append(y1)
        y1_fre.append(1)
    else:
        y1_fre[y1_points.index(y1)] += 1

    if y2 not in y2_points:
        y2_points.append(y2)
        y2_fre.append(1)
    else:
        y2_fre[y2_points.index(y2)] += 1

    max_x1_fre = max(x1_fre)
    max_x1_id = x1_points[x1_fre.index(max_x1_fre)]

    max_x2_fre = max(x2_fre)
    max_x2_id = x2_points[x2_fre.index(max_x2_fre)]

    max_y1_fre = max(y1_fre)
    max_y1_id = y1_points[y1_fre.index(max_y1_fre)]

    max_y2_fre = max(y2_fre)
    max_y2_id = y2_points[y2_fre.index(max_y2_fre)]


    return max_x1_id,max_x2_id,max_y1_id,max_y2_id


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    for line in lines:
        for x1,y1,x2,y2 in line:
            c = calc_angle(x1,y1,x2,y2)
            if c > 120 and c < 160:
                x1,x2,y1,y2 = stat_lines(x1,y1,x2,y2)
                #cv2.line(img,(x1,y1),(x2,y2), (0,255,0),4)
                cv2.circle(img, (x1,y1), 1, (0,0,255), 16)
                cv2.circle(img, (x2,y2), 1, (0,0,255), 16)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

def detect_yellow(img):
    #lower = np.array([0, 0, 180])
    #upper = np.array([48, 59, 255])
    #lower = np.array([0, 35, 175])
    #upper = np.array([48, 70, 255])
    lower = np.array([140, 0, 121])
    upper = np.array([180, 70, 255])
    #HSV限制
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    #二值化
    ret, binary = cv2.threshold(mask, 155, 255, cv2.THRESH_BINARY)
    #开运算
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #膨胀
    kernel = np.ones((15, 15), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    #霍夫变换
    rho = 2  # 设置极径分辨率
    theta = (np.pi) / 180  # 设置极角分辨率
    threshold = 700  # 设置检测一条直线所需最少的交点
    min_line_len = 200  # 设置线段最小长度
    max_line_gap = 50  # 设置线段最近俩点的距离
    lines = cv2.HoughLinesP(binary, rho, theta, threshold, np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap)
    hough_line_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    # 绘制检测到的直线
    draw_lines(hough_line_image, lines)
    sync_image = cv2.addWeighted(img, 0.8, hough_line_image, 1, 0)
    cv2.namedWindow("hh", cv2.WINDOW_NORMAL)
    cv2.imshow('hh', binary)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.imshow('mask', sync_image)

cap = cv2.VideoCapture('./2.mp4')
while True:
    ret, frame = cap.read()
    detect_yellow(frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()
