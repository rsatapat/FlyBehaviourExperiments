import cv2
import numpy as np
import image_methods

def fly_pos(contours,size_cutoff=60):
    pos = []
    for i in range(0, len(contours)):
        if len(contours[i]) > 5 and len(contours[i]) < 500:
            M = cv2.moments(contours[i])
            Area = cv2.contourArea(contours[i])
            if Area > size_cutoff:
                # print(Area)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pos.append([cx,cy])
                break
    return pos, Area

def fly_court_pos(contours,size_cutoff=200, max_size = 99999999):
    pos = []
    ellipse = [0,0,0]
    cnt = []
    for i in range(0, len(contours)):
        # print("length of contour {}".format(len(contours[i])))
        if len(contours[i]) > 15 and len(contours[i]) < 500:
            M = cv2.moments(contours[i])
            Area = cv2.contourArea(contours[i])
            if Area > size_cutoff and Area < max_size:
                # print(Area)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pos.append([cx, cy])
                cnt.append(contours[i])
                ellipse = cv2.fitEllipse(contours[i])
    return pos,ellipse,cnt

def fly_court_pos_new(contours,size_cutoff=60):
    pos = []
    ellipse = []
    cnt = []
    area = []
    for i in range(0, len(contours)):
        if len(contours[i]) > 15 and len(contours[i]) < 500:
            M = cv2.moments(contours[i])
            Area = cv2.contourArea(contours[i])
            if Area > size_cutoff:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pos.append([cx, cy])
                ellipse.append(cv2.fitEllipse(contours[i]))
                area.append(Area)
                cnt.append(contours[i])
    return pos,ellipse,cnt,area

def get_fly_postion_and_orientaion(camera, loc, size):
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_court_pos(contours, size_cutoff=200, max_size=4000)
    fly_ori = ellipse[2]

    return pos, fly_ori, timestamp

def ROI(image, x, y):
    height, width = image.shape
    loc1 = x
    size1 = y
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, tuple(loc1), size1, 1, thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data