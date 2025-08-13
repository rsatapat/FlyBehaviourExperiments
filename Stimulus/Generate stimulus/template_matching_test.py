import cv2
import numpy as np
import fly_track
i = 0
cap = cv2.VideoCapture(r'{}/{}'.format(r'D:\Roshan\Experiments\Courtship\WT_blind_9_9', 'WT_blind_9_9_2019826_corrected_vid.avi.avi'))
cap.set(cv2.CAP_PROP_POS_FRAMES, 43200+900+280)
while (cap.isOpened()):
    ret, image = cap.read()
    # image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()))
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    ret, diff_img = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY_INV)


    # cv2.circle(diff_img,tuple(cnt[point][0]),(2,30),(ellipse[0][2]+90),0,360,[0,0,0],-1)
    # opening = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, kernel, iterations=2)
    #
    # # sure background area
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(diff_img, cv2.DIST_C,0)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # # unknown = cv2.subtract(sure_bg, dist_transform)

    # this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # pos, ellipse, cnt, area = fly_track.fly_court_pos_new(contours)

    # img_erosion = cv2.erode(diff_img, kernel, iterations=1)
    # img_dilation = cv2.dilate(diff_img, kernel, iterations=1)

    # cv2.drawContours(image, [hull_cnt], 0, (255, 0, 0), 1)
    # cv2.imshow('frame', np.concatenate((img_erosion, img_dilation),axis=1))
    cv2.imshow('frame',np.concatenate((image, diff_img),axis=1))


    i = i + 1
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

