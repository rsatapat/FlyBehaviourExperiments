import cv2
import numpy as np
import image_methods
import math
import matplotlib.pyplot as plt

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

def fly_court_pos_new(contours, size_cutoff=60):
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

    return image, pos, fly_ori, timestamp

def get_male_female_postion_and_orientaion(camera, loc, size, last_position, last_ori, female_max_cutoff, female_min_cutoff, 
                                           male_max_cutoff, male_min_cutoff, double_min_cutoff, flies_touching, fly_lost):
    position = last_position
    ori = last_ori
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY_INV)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt, area = fly_court_pos_new(contours, size_cutoff=male_min_cutoff)
    
    ##
    if len(pos) == 2: ## and flies were not touching in the last frame
                        ## or know the position of flies with high certainty
        dist1 = math.sqrt((pos[0][0] - last_position[0][0]) ** 2 + (pos[0][1] - last_position[0][1]) ** 2)
        dist2 = math.sqrt((pos[1][0] - last_position[0][0]) ** 2 + (pos[1][1] - last_position[0][1]) ** 2)
        dist3 = math.sqrt((pos[0][0] - last_position[1][0]) ** 2 + (pos[0][1] - last_position[1][1]) ** 2)
        dist4 = math.sqrt((pos[1][0] - last_position[1][0]) ** 2 + (pos[1][1] - last_position[1][1]) ** 2)
        if flies_touching == 0 and fly_lost == 0:
            dist_sorted = [dist1, dist2, dist3, dist4]
            dist_sorted.sort()
            flies_touching = 0
            if dist_sorted[:2] in [[dist1, dist4], [dist4, dist1]]:
                position = [pos[0], pos[1]]
                ori = [ellipse[0][2], ellipse[1][2]]
                male_area = area[0]
                female_area = area[1]
                # fly_contours = [cnt[0],cnt[1]]

            elif dist_sorted[:2] in [[dist2, dist3], [dist3, dist2]]:
                position = [pos[1], pos[0]]
                ori = [ellipse[1][2], ellipse[0][2]]
                male_area = area[1]
                female_area = area[0]
                # fly_contours = [cnt[1], cnt[0]]
            else:
                ##use area
                # print('using area')
                if area[1] < female_max_cutoff and area[1] > female_min_cutoff and area[0] < male_max_cutoff and area[0] > male_min_cutoff:
                    position = [pos[0], pos[1]]
                    ori = [ellipse[0][2], ellipse[1][2]]
                    male_area = area[0]
                    female_area = area[1]
                    # fly_contours = [cnt[0],cnt[1]]

                elif area[0] < female_max_cutoff and area[0] > female_min_cutoff and area[1] < male_max_cutoff and area[1] > male_min_cutoff:
                    position = [pos[1], pos[0]]
                    ori = [ellipse[1][2], ellipse[0][2]]
                    male_area = area[1]
                    female_area = area[0]
                    # fly_contours = [cnt[1], cnt[0]]
                else:
                    print('Both flies not detected by size...')
                    print(area[0],area[1])
                    position = [last_position[0], last_position[1]]
                    ori = [last_ori[0], last_ori[1]]

        else:
            ## use area
            if area[0] < area[1]:
                position = [pos[0], pos[1]]
                ori = [ellipse[0][2], ellipse[1][2]]
                # fly_contours = [cnt[0], cnt[1]]
            else:
                position = [pos[1], pos[0]]
                ori = [ellipse[1][2], ellipse[0][2]]
                # fly_contours = [cnt[1], cnt[0]]
        # ours = [cnt[0], cnt[1]]
            flies_touching = 0

    elif len(pos)<2:
        if len(pos) == 0:
            pos.append(position[0])
            pos.append(position[1])

        elif len(pos) == 1:
            if area[0] > double_min_cutoff:
                print('Flies are in contact...', area[0])
                flies_touching = 1
                # position = last_position
                # ori = [las_ori[0],last_ori[1]]
                position = [pos[0], pos[0]]
                ori = [last_ori[0], last_ori[1]]

            elif area[0] < female_max_cutoff and area[0] > female_min_cutoff:
                pos.insert(0,position[0])
                ellipse.insert(0, [0, 0,last_ori[0]])
                # cnt.insert(0,cnt[0])
                position = [pos[0], pos[1]]
                ori = [ellipse[0][2], ellipse[1][2]]
                # fly_contours = [cnt[0], cnt[1]]

            elif area[0] < male_max_cutoff and area[0] > male_min_cutoff:
                pos.append(position[1])
                ellipse.append([0,0,last_ori[1]])
                # cnt.append(cnt[0])
                position = [pos[0], pos[1]]
                ori = [ellipse[0][2], ellipse[1][2]]
                # fly_contours = [cnt[0], cnt[1]]
            else:
                # print(area[0])
                position = [last_position[0], last_position[1]]
                ori = [last_ori[0], last_ori[1]]
                # position = [pos[0], pos[1]]
                # ori = [ellipse[0][2], ellipse[1][2]]
                # fly_contours = [cnt[0], cnt[1]]

    elif len(pos)>2:
        if math.sqrt((pos[0][1]-pos[1][1])**2 + (pos[0][0]-pos[1][0])**2)<10:
            print('Error : Rogue fly detection...')
    
    male_position = position[0]
    female_position = position[1]
    male_fly_ori = ori[0]
    female_fly_ori = ori[1]
    distance_between_flies = math.sqrt((position[0][0] - position[1][0]) ** 2 + (position[0][1] - position[1][1]) ** 2)
    ##
    return image, male_position, female_position, male_fly_ori, female_fly_ori, distance_between_flies, timestamp, area, flies_touching, fly_lost

def ROI(image, x, y):
    height, width = image.shape
    loc1 = x
    size1 = y
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, tuple(loc1), size1, 1, thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data


def find_size(camera, loc, size):
    frame = 0
    large_area = []
    small_area = []
    while (True):
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        # image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY_INV)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt, area = fly_court_pos_new(contours)

        if len(pos) != 2:
            print(len(pos))
            print('wrong')
            continue
        else:
            area.sort()
            large_area.append(area[1])
            small_area.append(area[0])
        frame = frame + 1
        cv2.imshow('frame',cv_image)
        cv2.waitKey(1)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    print('Male fly: max{},min{}'.format(max(small_area),min(small_area)))
    print('Female fly: max{},min{}'.format(max(large_area),min(large_area)))
    plt.hist(small_area,bins= np.linspace(300,800,101),color=(1,0,0), alpha = 0.6)
    plt.hist(large_area,bins= np.linspace(300,800,101),color =(0,1,0),alpha = 0.6)
    plt.show()

    cv2.destroyAllWindows()
    return small_area, large_area
    # return [max(small_area),min(small_area)], [max(large_area),min(large_area)]


def find_male(camera, cutoff, loc, size):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    male_x = 0
    male_y = 0
    last_position = [x1,y1,x2,y2]
    while(True):
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        # image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY_INV)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt, _ = fly_court_pos_new(contours)

        if len(pos)<2:
            if len(pos) == 0:
                continue
            elif len(pos) == 1:
                # if cv2.contourArea(cnt[0]) > 400:
                #     hull = cv2.convexHull(cnt[0],returnPoints = False)
                #     defects = cv2.convexityDefects(cnt[0],hull)
                #     max_dist = 0
                #     for j in range(defects.shape[0]):
                #         if max_dist < defects[j,0,3]:
                #             max_dist = defects[j,0,3]
                #     index = np.where(defects == max_dist)
                #     index = defects[index[0][0],0, 2]
                #     cv2.ellipse(diff_img,tuple(cnt[index][0]),(2,30),(ellipse[0][2]+90),0,360,[0,0,0],-1)
                #     this,contours,hierarchy = cv2.findContours(diff_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                #
                #     pos, ellipse, cnt,_ = fly_track.fly_court_pos_new(contours)
                # else:
                #     continue
                continue
            else:
                continue

        elif len(pos)>2:
            if math.sqrt((pos[0][1]-pos[1][1])**2 + (pos[0][0]-pos[1][0])**2)<10:
                print('Error : Rogue fly detection...')
                continue
            
        cv2.imshow('frame1',diff_img)
        cv2.waitKey(1)
        #position
        try:
            dist1 = (pos[0][0]-x1)**2 + (pos[0][1]-y1)**2
            dist2 = (pos[1][0]-x1)**2 + (pos[1][1]-y1)**2
        except:
            cv2.ellipse(diff_img,tuple(cnt[index][0]),(5,30),(ellipse[0][2]+90),0,360,[1,1,0],-1)
            cv2.imshow('frame',diff_img)
         
        if dist1 < dist2:
            x1 = pos[0][0]
            y1 = pos[0][1]
            x2 = pos[1][0]
            y2 = pos[1][1]
        else:
            # print('switch')
            x1 = pos[1][0]
            y1 = pos[1][1]
            x2 = pos[0][0]
            y2 = pos[0][1]

        cv2.waitKey(1)
        if cv2.waitKey(100) & 0xFF == ord('m'):
            print('Is this the male?')
            if male_x == last_position[0]:
                male_x = x2
                male_y = y2
                female_x = x1
                female_y = y1
            else:
                male_x = x1
                male_y = y1
                female_x = x2
                female_y = y2
        elif cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return male_x, male_y, female_x, female_y
        else:
            if male_x == last_position[0]:
                male_x = x1
                male_y = y1
                female_x = x2
                female_y = y2
            else:
                male_x = x2
                male_y = y2
                female_x = x1
                female_y = y1
        cv2.circle(cv_image,(int(male_x),int(male_y)), 10, (255,255,255), 2)
        cv2.imshow('frame',cv_image)
        last_position = [x1,y1,x2,y2]