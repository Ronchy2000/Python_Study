import cv2 as cv
import numpy as  np


color_case_r ,color_case_b, red_area , blue_area=0,0,0,0
def detect_color_area(img , max_area):
    global color_case_r ,color_case_b, red_area , blue_area
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_hsv_red_1 = np.array([0, 43, 46])
    upper_hsv_red_1 = np.array([10, 255, 255])

    lower_hsv_red_2 = np.array([156, 43, 46])
    upper_hsv_red_2 = np.array([180, 255, 255])

    lower_hsv_blue = np.array([100, 43, 46])
    upper_hsv_blue = np.array([124, 255, 255])

    mask_red_1 = cv.inRange(hsv, lowerb=lower_hsv_red_1, upperb=upper_hsv_red_1)
    mask_red_2 = cv.inRange(hsv, lowerb=lower_hsv_red_2, upperb=upper_hsv_red_2)

    frame_threshold_red = mask_red_1 + mask_red_2

    frame_threshold_blue = cv.inRange(hsv, lowerb=lower_hsv_blue, upperb=upper_hsv_blue)
    #
    #cv.imwrite("blue.jpg", frame_threshold_blue)
    #cv.imwrite("red.jpg", frame_threshold_red )
    #
    # # 3.先腐蚀，后膨胀去除噪声，然后过度膨胀，放大目标。
    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>腐蚀<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    kernel_1 = np.ones((2, 2), np.uint8)  # 定义卷积核

    erosion_blue = cv.erode(frame_threshold_blue, kernel_1, iterations=2)

    kernel_2 = np.ones((2, 2), np.uint8)

    dilation_blue = cv.dilate(erosion_blue, kernel_2, iterations=2)

    kernel_1 = np.ones((2, 2), np.uint8)  # 定义卷积核

    erosion_red = cv.erode(frame_threshold_red, kernel_1, iterations=2)

    kernel_2 = np.ones((2,2), np.uint8)

    dilation_red = cv.dilate(erosion_red , kernel_2, iterations=2)

    color_case = '0000'
    red_area   = '0000'
    blue_area  = '0000'


    #>>>>>>>>>>>>>>>>>>>找红色色块<<<<<<<<<<<<<<<<<<<<<<

    contours_red , hierarchy = cv.findContours(dilation_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #print("红色色块数量：" , len(contours_red))

    red_area_list = []

    red_blob_count = len(contours_red)

    if len(contours_red) != 0:

        for cnt in range(len(contours_red)):  # 对检测到的每个轮廓遍历
            # p = cv.arcLength(contours[cnt], True)
            area = cv.contourArea(contours_red[cnt])  # area是该轮廓的像素面积

            red_area_list.append(area)

        red_area_list.sort()

        Max_red = red_area_list[0]

        if Max_red > max_area:
            red_area = Max_red

            #print("the red blob max :" , Max_red)
            #print("已返回")

        else:

            red_area = "None"

    elif len(contours_red) == 0 :
        red_area = 0


    # >>>>>>>>>>>>>>>>>>>找蓝色色块<<<<<<<<<<<<<<<<<<<<<<

    contours_blue , hierarchy = cv.findContours(dilation_blue , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #print("蓝色色块数量：", len(contours_blue))

    blue_area_list = []
    blue_blob_count = len(contours_blue)

    if len(contours_blue) != 0:

        for cnt in range(len(contours_blue)):  # 对检测到的每个轮廓遍历
            # p = cv.arcLength(contours[cnt], True)  # p是Perimeter周长的意思，当时偷懒了
            area = cv.contourArea(contours_blue[cnt])  # area是该轮廓的像素面积

            blue_area_list.append(area)

        blue_area_list.sort()

        Max_blue = blue_area_list[0]

        if Max_blue > max_area:
            blue_area = Max_blue

           # print("the blue blob max :", Max_blue)
           # print("已返回")

        else:
            blue_area = "None"

    elif len(contours_blue) == 0:
        blue_area = 0



    if red_blob_count ==0 and blue_blob_count == 0:
        color_case = 1
    elif red_blob_count != 0 and blue_blob_count == 0:
        color_case_r = 1			#只有红色
    elif blue_blob_count != 0 and red_blob_count == 0:
        color_case_b = 1			#只有蓝色


    return color_case_r ,color_case_b, red_area , blue_area

if __name__ == '__main__':
   img = cv.imread('E:\Developer\Python\Myworkshop\Python_Study\Opencv\helloCV_OIP.jpg')

   color_case1,color_case2 , red_area1 , blue_area1 = detect_color_area(img , 200)
   print(color_case1 , color_case2 , red_area1 , blue_area1)


