import copy
import time
import random
import os
import datetime
import math
import csv
import json
import pickle
import cv2
import numpy as np

import psychopy.event
import psychopy.visual
import skvideo.io as sk
import shutil

# my modules
from Dialog_box import info
from File_chooser import file_chooser
import objects
import fly_track
import image_methods
import misc
import message_box

##function for generating mask
def ROI(image, x, y):
    height, width = image.shape
    loc1 = x
    size1 = y
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, tuple(loc1), size1, 1, thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data

##camera intialisation
camera = image_methods.initialise_camera("ptgrey")
# camera = image_methods.initialise_camera("basler")]

pos_win=[2160, -40]
win = misc.initialise_projector_window(r'D:/', pos=pos_win)

# circles_found = []
# i = 0
# while i < 10:
#     image, timestamp = image_methods.grab_image(camera, 'ptgrey')
#     circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 200,
#                                param1=150, param2=50, minRadius=65, maxRadius=75)
#     if circles is None:
#         pass
#     else:
#         circles_found.append(circles[0, 0, :2])
#     i += 1
# 
# # print(circles_found)
# print(np.median(circles_found, axis=0))

w_ratio = 500
w_rem = 0
h_ratio = 500
h_rem = 0

image, timestamp = image_methods.grab_image(camera, 'ptgrey')
height, width = image.shape
arena = cv2.selectROI(image, False)
loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
size = arena[2] // 2

circle1 = psychopy.visual.Circle(
    win = win,
    radius = 0.2,
    lineColor = (-1, -1, -1),
    fillColor = (-1, -1, -1)
)
circle2 = psychopy.visual.Circle(
    win = win,
    radius = 0.2,
    lineColor = (-1, -1, -1),
    fillColor = (-1, -1, -1)
)
circle3 = psychopy.visual.Circle(
    win = win,
    radius = 0.2,
    lineColor = (-1, -1, -1),
    fillColor = (-1, -1, -1)
)
circle4 = psychopy.visual.Circle(
    win = win,
    radius = 0.2,
    lineColor = (-1, -1, -1),
    fillColor = (-1, -1, -1)
)

while(True):
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow('frame', diff_img)
    cv2.waitKey(0)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pos = []
    ellipse = [0, 0, 0]
    cnt = []
    for i in range(0, len(contours)):
        # print("length of contour {}".format(len(contours[i])))
        if len(contours[i]) > 15 and len(contours[i]) < 500:
            M = cv2.moments(contours[i])
            Area = cv2.contourArea(contours[i])
            if Area > 500 and Area < 30000:
                # print(Area)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pos.append([cx, cy])
                cnt.append(contours[i])
                ellipse = cv2.fitEllipse(contours[i])
    print(pos)
    x = (pos[0][0] - 512) / w_ratio + w_rem
    y = (pos[0][1] - 512) / h_ratio + h_rem
    circle1.pos = [[x, y]]

    x = (pos[1][0] - 512) / w_ratio + w_rem
    y = (pos[1][1] - 512) / h_ratio + h_rem
    circle2.pos = [[x, y]]

    x = (pos[2][0] - 512) / w_ratio + w_rem
    y = (pos[2][1] - 512) / h_ratio + h_rem
    circle3.pos = [[x, y]]

    x = (pos[3][0] - 512) / w_ratio + w_rem
    y = (pos[3][1] - 512) / h_ratio + h_rem
    circle4.pos = [[x, y]]

    # cv2.imshow('frame', image[:128, :128])
    input = cv2.waitKeyEx(0)
    print(pos_win, input)
    if input == 32:
        pass
    elif input == 2424832:
        pos_win[0] = pos_win[0] - 2
        print(pos_win)
        cv2.destroyAllWindows()
        win = misc.initialise_projector_window(r'D:/', pos_win)
    elif input == 2490368:
        pos_win[1] = pos_win[1] + 2
        cv2.destroyAllWindows()
        win = misc.initialise_projector_window(r'D:/', pos_win)
    elif input == 2555904:
        pos_win[0] = pos_win[0] + 2
        cv2.destroyAllWindows()
        win = misc.initialise_projector_window(r'D:/', pos_win)
    elif input == 2621440:
        pos_win[1] = pos_win[1] - 2
        cv2.destroyAllWindows()
        win = misc.initialise_projector_window(r'D:/', pos_win)
    else:
        pass
    circle1.win = win
    circle2.win = win
    circle3.win = win
    circle4.win = win
    circle1.draw()
    circle2.draw()
    circle3.draw()
    circle4.draw()
    win.flip()
