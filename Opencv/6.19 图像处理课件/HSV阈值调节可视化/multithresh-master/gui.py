import cv2
import numpy as np
import json
import os


def nothing(x):
    pass


setting = {'hue_min': 0, 'hue_max': 180, 'sat_min': 0, 'sat_max': 255, 'val_min': 0, 'val_max': 255, 'on': 0}
setting_file = os.path.join(os.path.expanduser('~'), '.multithresh.json')
if os.path.exists(setting_file):
    with open(setting_file, 'r') as f:
        setting = json.load(f)

winname = "multithresh"
switch = '0 : OFF \n1 : ON'
img = np.zeros((300,512,3), np.uint8)

cv2.namedWindow(winname)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow(winname, img)
cv2.createTrackbar('H_min', winname, setting['hue_min'], 180, nothing)
cv2.createTrackbar('H_max', winname, setting['hue_max'], 180, nothing)
cv2.createTrackbar('S_min', winname, setting['sat_min'], 255, nothing)
cv2.createTrackbar('S_max', winname, setting['sat_max'], 255, nothing)
cv2.createTrackbar('V_min', winname, setting['val_min'], 255, nothing)
cv2.createTrackbar('V_max', winname, setting['val_max'], 255, nothing)
cv2.createTrackbar(switch, winname, setting['on'], 1, nothing)
cv2.imshow(winname, img)


def refresh():
    global setting
    setting['hue_min'] = cv2.getTrackbarPos('H_min', winname)
    setting['hue_max'] = cv2.getTrackbarPos('H_max', winname)
    setting['sat_min'] = cv2.getTrackbarPos('S_min', winname)
    setting['sat_max'] = cv2.getTrackbarPos('S_max', winname)
    setting['val_min'] = cv2.getTrackbarPos('V_min', winname)
    setting['val_max'] = cv2.getTrackbarPos('V_max', winname)
    setting['on'] = cv2.getTrackbarPos(switch, winname)
    print("-"*50)
    print(f"Hue: {setting['hue_min']*2:d}-{setting['hue_max']*2:d} deg")
    print(f"Sat: {setting['sat_min']/2.55:.0f}-{setting['sat_max']/2.55:.0f} %")
    print(f"Val: {setting['val_min']/2.55:.0f}-{setting['val_max']/2.55:.0f} %")

    key = cv2.waitKey(50)
    if key == 27:
        raise KeyboardInterrupt()


def display(image):
    global setting
    while image.shape[0] * image.shape[1] > 500000:
        image = cv2.pyrDown(image)
    if setting['on'] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(image, (setting['hue_min'], setting['sat_min'], setting['val_min']),
                             (setting['hue_max'], setting['sat_max'], setting['val_max']))
        cv2.imshow("Image", thresh)
    else:
        cv2.imshow("Image", image)


def close():
    with open(setting_file, 'w') as f:
        json.dump(setting, f)
    cv2.destroyWindow(winname)
    cv2.destroyWindow("Image")
