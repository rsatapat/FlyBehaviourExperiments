import copy
import time
import random
import os
import math
import datetime
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
from Dialog_box_courtship import info
from File_chooser import file_chooser
import objects
import fly_track
import image_methods
import matplotlib.pyplot as plt
import misc
import scipy.spatial


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def find_size(camera):
    frame = 0
    large_area = []
    small_area = []
    while (True):
        # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 55, 255, cv2.THRESH_BINARY_INV)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pos, ellipse, cnt, area = fly_track.fly_court_pos_new(contours)

        if len(pos) != 2:
            print(len(pos))
            print('wrong')
            continue
        else:
            area.sort()
            large_area.append(area[1])
            small_area.append(area[0])
        frame = frame + 1
        cv2.imshow('frame', cv_image)
        cv2.waitKey(1)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    print('Male fly: max{},min{}'.format(max(small_area), min(small_area)))
    print('Female fly: max{},min{}'.format(max(large_area), min(large_area)))
    plt.hist(small_area, bins=np.linspace(300, 800, 101), color=(1, 0, 0), alpha=0.6)
    plt.hist(large_area, bins=np.linspace(300, 800, 101), color=(0, 1, 0), alpha=0.6)
    plt.show()

    proceed = input('Do we proceed?: ')
    print('you said {}', format(proceed))
    if proceed == 'y' or 'Y' or 'Yes' or 'yes':
        plt.close()
        pass
    else:
        plt.close()
        small_area, large_area = find_size(camera)
    return small_area, large_area
    # return [max(small_area),min(small_area)], [max(large_area),min(large_area)]


def find_male(camera, cutoff):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    male_x = 0
    male_y = 0
    last_position = [x1, y1, x2, y2]
    while (True):
        # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 55, 255, cv2.THRESH_BINARY_INV)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pos, ellipse, cnt, _ = fly_track.fly_court_pos_new(contours)

        if len(pos) < 2:
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

        elif len(pos) > 2:
            if math.sqrt((pos[0][1] - pos[1][1]) ** 2 + (pos[0][0] - pos[1][0]) ** 2) < 10:
                print('Error : Rogue fly detection...')
                continue

        cv2.imshow('frame1', diff_img)
        cv2.waitKey(1)
        # position
        try:
            dist1 = (pos[0][0] - x1) ** 2 + (pos[0][1] - y1) ** 2
            dist2 = (pos[1][0] - x1) ** 2 + (pos[1][1] - y1) ** 2
        except:
            cv2.ellipse(diff_img, tuple(cnt[index][0]), (5, 30), (ellipse[0][2] + 90), 0, 360, [1, 1, 0], -1)
            cv2.imshow('frame', diff_img)

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
        cv2.circle(cv_image, (int(male_x), int(male_y)), 10, (255, 255, 255), 2)
        cv2.imshow('frame', cv_image)
        last_position = [x1, y1, x2, y2]


if __name__ == "__main__":
    exp_info = info()

    # fly_parameters
    male_strain = exp_info[0]
    female_strain = exp_info[1]
    male_age = int(exp_info[2])
    female_age = int(exp_info[3])
    male_serial = int(exp_info[4])
    female_serial = int(exp_info[5])
    minutes = int(exp_info[6])
    fly_M = objects.fly(male_strain, male_serial, "M", male_age)  ## male fly object defined here
    fly_F = objects.fly(female_strain, female_serial, "F", female_age)  ## female fly object defined here
    if exp_info[6] == '_':
        remark = ''
    else:
        remark = exp_info[6]

    # experiment parameters
    bg = 1
    fg = -1
    reps = int(exp_info[6])
    size_stim = [0.25, 0.5, 0.75, 1.0]
    bright_bars = np.linspace(0.0, 1.0, 11)
    dark_bars = np.linspace(0.0, 1.0, 11)
    date_time = datetime.datetime.now()
    num_bars = [6]
    freq = [10]
    stimulus = 'pinwheel'
    if exp_info[7] == '_':
        remark = ''
    else:
        remark = exp_info[6]

    # video parameters
    framerate = 60
    resolution = (1024, 1024)
    vid = objects.video(resolution, framerate)  ## vid object defined here

    Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
    if os.path.isdir(Dir) == False:  ## checks if this folder exists
        print('Folder name incorrect')
        exit()

    date_string = ''
    for i in [date_time.year, date_time.month, date_time.day]:
        date_string = date_string + str(i)

    filename = male_strain + '_' + female_strain + '_' + str(male_serial) + '_' + str(female_serial) + '_' + date_string
    Dir_name = male_strain + '_' + female_strain + '_' + str(male_serial) + '_' + str(female_serial)
    Dir_new = r'{}\{}'.format(Dir, Dir_name)
    if os.path.isdir(Dir_new) == False:  ## checking to see of the fly already has a folder or not
        os.mkdir(Dir_new)  ## if no such folder exists, make one

    file_name_json = filename + '.json'
    file_name_pickle = filename + '.p'

    files_path_json = r'{}\{}'.format(Dir_new, file_name_json)
    files_path_pickle = r'{}\{}'.format(Dir_new, file_name_pickle)
    f_pickle = open(files_path_pickle, 'ab')

    filename_vid = filename + '_' + remark
    outpath = r'{}\{}'.format(Dir_new, filename_vid)
    csv_file = outpath + '.csv'
    csv_file = open(csv_file, 'w+')
    writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer_csv.writerow(
        ['pos_x_m', 'pos_y_m', 'ori_m', 'pos_x_f', 'pos_y_f', 'ori_f', 'timestamp', 'direction', 'confidence'])
    writer_small_male = sk.FFmpegWriter(outpath + 'male' + '.avi', inputdict={'-r': '60'},
                                        outputdict={'-vcodec': 'libx264',
                                                    '-crf': '0',
                                                    '-preset': 'slow'})
    writer_small_female = sk.FFmpegWriter(outpath + 'female' + '.avi', inputdict={'-r': '60'},
                                          outputdict={'-vcodec': 'libx264',
                                                      '-crf': '0',
                                                      '-preset': 'slow'})
    writer_small = sk.FFmpegWriter(outpath + '.avi', inputdict={'-r': '150'}, outputdict={'-vcodec': 'libx264',
                                                                                          '-crf': '0',
                                                                                          '-preset': 'slow'})

    ## experiment object defined here
    experiment = objects.exp([date_time.year, date_time.month, date_time.day],
                             [date_time.hour, date_time.minute, date_time.second, date_time.microsecond], bg, fg,
                             filename_vid)

    total_data = {**{**fly_M.__dict__, **fly_F.__dict__}, **{'exp_params': []}}  # fly_data added to the dictionary

    f_json = open(files_path_json,
                  'w')  ## open the json file ## important : data is not appended but re-written at the end of the experiment
    pickle.dump({**fly_M.__dict__, **fly_F.__dict__},
                f_pickle)  ## start writing to the pickle file, it acts as a temporary storage, in case of ant failure in writing to the json file

    total_data['exp_params'].append(
        {**vid.__dict__, **experiment.__dict__, **{'stim_params': []}})  ## experiment data added to the dictionary
    pickle.dump({**vid.__dict__, **experiment.__dict__}, f_pickle)

    ##camera intialisation
    # camera = image_methods.initialise_camera("ptgrey")
    camera = image_methods.initialise_camera("basler")

    win = misc.initialise_projector_window(Dir)

    ##psychopy stimulus window initialisation
    win = psychopy.visual.Window(
        size=[342, 684],
        # pos = [1998,-20],
        pos=[1920, -20],
        color=(0, 0, 0),
        fullscr=False,
        waitBlanking=False
    )


    bars = random.choice(num_bars)
    Hz = random.choice(freq)
    size = random.choice(size_stim)
    contrast1 = 100
    contrast2 = 75
    contrast3 = 50
    contrast4 = 25
    contrast5 = 10
    angle = (360 * Hz) / (framerate * bars)
    print(bars, angle)

    # a rectangle to draw all over the arena to replicate light and dark conditions
    rect_stim_light = psychopy.visual.Rect(
        win=win,
        units='norm',
        width=16,
        height=2,
        lineColor=-0.9,
        fillColor=-0.9,
        pos=(0, 0)
    )
    rect_stim_light.autoDraw = False

    # random dot stimulus to have rotation with dots, not gratings
    xys = []
    for i in np.linspace(-0.75, 0.75, 5):
        for j in np.linspace(-0.75, 0.75, 5):
            xys.append([i, j])
    random_dot_stim = psychopy.visual.ElementArrayStim(
        win=win,
        units='norm',
        fieldSize=(2, 2),
        nElements=25,
        xys=xys,
        sizes=1 / 4,
        colors=-1
    )
    random_dot_stim.autoDraw = False

    # dot stimulus to be used as a female
    dot_stim = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1 / 512,
        pos=(0, 0),
        lineColor=0
    )
    dot_stim.autoDraw = False

    w_ratio = 512
    w_rem = 0
    h_ratio = 512
    h_rem = 0

    ##first image taken for pre-processing
    # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    image, timestamp = image_methods.grab_image(camera, 'basler')
    height, width = image.shape

    ##user finds the arena, by selecting ROI
    arena = cv2.selectROI(image, False)
    loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
    size = arena[2] // 2

    small_fly, large_fly = find_size(camera)

    fig1 = plt.figure()
    plt.hist(small_fly)
    plt.hist(large_fly)
    fig1.savefig('{}/{}'.format(Dir, 'large_fly.png'))
    print(max(small_fly), min(small_fly))
    print(max(large_fly), min(large_fly))

    male_min_cutoff = min(small_fly) - 30
    female_min_cutoff = min(large_fly)
    male_max_cutoff = max(small_fly)
    female_max_cutoff = max(large_fly) + 30
    double_min_cutoff = male_min_cutoff + female_min_cutoff - 50

    print(male_min_cutoff, male_max_cutoff, female_min_cutoff, female_max_cutoff, double_min_cutoff)

    cv_image = ROI(image, loc, size)

    fly_pos = find_male(camera, 60)

    position = [[fly_pos[0], fly_pos[1]], [fly_pos[2], fly_pos[3]]]

    # bg_image = copy.deepcopy(cv_image)
    # bg_image[fly_pos[0]:fly_pos[0]+20,fly_pos[1]:fly_pos[1]+20] ### MAKE CHANGE HERE

    time_exp_start = time.clock()
    time_stim_start = time.clock()

    direction = random.choice([0, 1, -1])
    filler_frame = np.zeros((100, 100))
    frame_number = 0
    stim_status = 0
    stim = 1
    stim_time = random.choice([3])
    video_frame = 0
    courtship_start_point = 0
    cv2.destroyAllWindows()
    contrast = random.choice([50, 75, 100])
    confidence = 1
    last_position = position
    last_ori = [0, 0]
    # fly_contours = [0,0]
    print('starting loops')
    print(video_frame)
    # overlap_polygon = [[0, 0], [0, 1], [1, 1]]
    # overlap_polygon = np.array(overlap_polygon)
    flies_touching = 0
    fly_lost = 0
    male_area = 0
    female_area = 0

    for color in [0, -1, -0.25, -0.9, -0.5, -0.8]:
        rect_stim_light.fillColor = color
        rect_stim_light.draw()
        win.flip()
        print(rect_stim_light.fillColor)
        win.flip()
        print(random_dot_stim.colors)
        minutes = 5
        frame_number = 0
        stim_status = 0
        stim = 1
        video_frame = 0
        while (video_frame < minutes * 60 * framerate):
            # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
            image, timestamp = image_methods.grab_image(camera, 'basler')
            cv_image = ROI(image, loc, size)

            ret, diff_img = cv2.threshold(cv_image, 55, 255, cv2.THRESH_BINARY_INV)
            # cv2.polylines(diff_img, [overlap_polygon], True, (0,0,0))
            this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pos, ellipse, cnt, area = fly_track.fly_court_pos_new(contours)
            # print(len(pos))
            if len(pos) == 2:  ## and flies were not touching in the last frame
                ## or know the position of flies with high certainty
                # print('here')
                dist1 = math.sqrt((pos[0][0] - last_position[0][0]) ** 2 + (pos[0][1] - last_position[0][1]) ** 2)
                dist2 = math.sqrt((pos[1][0] - last_position[0][0]) ** 2 + (pos[1][1] - last_position[0][1]) ** 2)
                dist3 = math.sqrt((pos[0][0] - last_position[1][0]) ** 2 + (pos[0][1] - last_position[1][1]) ** 2)
                dist4 = math.sqrt((pos[1][0] - last_position[1][0]) ** 2 + (pos[1][1] - last_position[1][1]) ** 2)

                if flies_touching == 0 and fly_lost == 0:
                    dist_sorted = [dist1, dist2, dist3, dist4]
                    dist_sorted.sort()
                    flies_touching = 0
                    if dist_sorted[:2] in [[dist1, dist4], [dist4, dist1]]:
                        # print('here')
                        position = [pos[0], pos[1]]
                        ori = [ellipse[0][2], ellipse[1][2]]
                        male_area = area[0]
                        female_area = area[1]
                        # fly_contours = [cnt[0],cnt[1]]

                    elif dist_sorted[:2] in [[dist2, dist3], [dist3, dist2]]:
                        # print('there')
                        position = [pos[1], pos[0]]
                        ori = [ellipse[1][2], ellipse[0][2]]
                        male_area = area[1]
                        female_area = area[0]
                        # fly_contours = [cnt[1], cnt[0]]
                    else:
                        ##use area
                        print('using area')
                        if area[1] < female_max_cutoff and area[1] > female_min_cutoff and area[0] < male_max_cutoff and \
                                area[0] > male_min_cutoff:
                            position = [pos[0], pos[1]]
                            ori = [ellipse[0][2], ellipse[1][2]]
                            male_area = area[0]
                            female_area = area[1]
                            # fly_contours = [cnt[0],cnt[1]]

                        elif area[0] < female_max_cutoff and area[0] > female_min_cutoff and area[
                            1] < male_max_cutoff and area[1] > male_min_cutoff:
                            position = [pos[1], pos[0]]
                            ori = [ellipse[1][2], ellipse[0][2]]
                            male_area = area[1]
                            female_area = area[0]
                            # fly_contours = [cnt[1], cnt[0]]
                        else:
                            print('Both flies not detected by size...')
                            print(area[0], area[1])
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

            elif len(pos) < 2:
                if len(pos) == 0:
                    pos.append(position[0])
                    pos.append(position[1])

                elif len(pos) == 1:

                    if area[0] > double_min_cutoff:
                        print('Flies are in contact...')
                        flies_touching = 1
                        position = [pos[0], pos[0]]
                        ori = [last_ori[0], last_ori[1]]
                    elif area[0] < female_max_cutoff and area[0] > female_min_cutoff:
                        pos.insert(0, position[0])
                        ellipse.insert(0, [0, 0, last_ori[0]])
                        # cnt.insert(0,cnt[0])
                        position = [pos[0], pos[1]]
                        ori = [ellipse[0][2], ellipse[1][2]]
                        # fly_contours = [cnt[0], cnt[1]]

                    elif area[0] < male_max_cutoff and area[0] > male_min_cutoff:
                        pos.append(position[1])
                        ellipse.append([0, 0, last_ori[1]])
                        # cnt.append(cnt[0])
                        position = [pos[0], pos[1]]
                        ori = [ellipse[0][2], ellipse[1][2]]
                        # fly_contours = [cnt[0], cnt[1]]
                    else:
                        print(area[0])
                        position = [last_position[0], last_position[1]]
                        ori = [last_ori[0], last_ori[1]]
                        # position = [pos[0], pos[1]]
                        # ori = [ellipse[0][2], ellipse[1][2]]
                        # fly_contours = [cnt[0], cnt[1]]

            elif len(pos) > 2:
                if math.sqrt((pos[0][1] - pos[1][1]) ** 2 + (pos[0][0] - pos[1][0]) ** 2) < 10:
                    print('Error : Rogue fly detection...')
                    continue

            distance_between_flies = math.sqrt(
                (position[0][0] - position[1][0]) ** 2 + (position[0][1] - position[1][1]) ** 2)

            writer_csv.writerow(
                [position[0][0], position[0][1], ori[0], position[1][0], position[1][1], ori[1], timestamp,
                 stim * contrast])

            x_m = (position[0][0] - 512) / w_ratio + w_rem
            y_m = (position[0][1] - 512) / h_ratio + h_rem
            x_f = (position[1][0] - 512) / w_ratio + w_rem
            y_f = (position[1][1] - 512) / h_ratio + h_rem

            wheel_stim100.pos = [[x_m, y_m]]
            wheel_stim75.pos = [[x_m, y_m]]
            wheel_stim50.pos = [[x_m, y_m]]
            # dot_stim.pos = [[x_m, y_m]]
            if frame_number < stim_time * framerate:
                if stim == 0:
                    # print('no stimulus')
                    pass
                else:
                    if contrast == 100:
                        wheel_stim100.ori = wheel_stim100.ori + direction * angle
                        wheel_stim100.draw()
                        stim_status = 100
                        # dot_stim.radius = dot_stim.radius + (Hz / 512)
                        # dot_stim.draw()
                        #      print(dot_stim.radius)

                    elif contrast == 75:
                        wheel_stim75.ori = wheel_stim75.ori + direction * angle
                        wheel_stim75.draw()
                        # dot_stim.radius = dot_stim.radius + (Hz / 512)
                        # dot_stim.draw()
                        stim_status = 75
                    elif contrast == 50:
                        wheel_stim50.ori = wheel_stim50.ori + direction * angle
                        wheel_stim50.draw()
                        # dot_stim.radius = dot_stim.radius + (Hz / 512)
                        # dot_stim.draw()
                        stim_status = 50
                # else:
                #     dot_stim.radius = dot_stim.radius + (Hz / 512)
            elif frame_number == stim_time * framerate:
                stim_status = 0
                STIM_ON_OFF = [0, 1]
                stimulus_inst = objects.stim_inst([stimulus, courtship_start_point], direction, bars, Hz,
                                                  time.clock() - time_exp_start,
                                                  time.clock() - time_stim_start,
                                                  courtship_start_point, video_frame, contrast, size)
                exp_number = len(total_data['exp_params'])
                total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
                pickle.dump(stimulus_inst.__dict__, f_pickle)
                stim = random.choice(STIM_ON_OFF)
                contrast = random.choice([50, 75, 100])
                direction = random.choice([0, 1, -1])
                Hz = random.choice([4, 8, 12])
                stim_time = random.choice([3])
                frame_number = 999999
                time_stim_start = time.clock()
                print('stimulus status {}'.format(stim))
                angle = (360 * Hz) / (framerate * bars)

            else:
                pass
            win.flip()

            if video_frame < courtship_start_point + ((stim_time * framerate) + 1):
                pass
            else:
                if frame_number % 1800 == 0:
                    frame_number = -1
                    stim = 1
                    courtship_start_point = video_frame
            cropped_image_male = cv_image[position[0][1] - 50:position[0][1] + 50,
                                 position[0][0] - 50:position[0][0] + 50]
            cropped_image_female = cv_image[position[1][1] - 50:position[1][1] + 50,
                                   position[1][0] - 50:position[1][0] + 50]

            ## saving video for male
            if cropped_image_male.shape == (100, 100):
                writer_small_male.writeFrame(cropped_image_male)
            else:
                writer_small_male.writeFrame(filler_frame)

            ## saving video for female
            if cropped_image_female.shape == (100, 100):
                writer_small_female.writeFrame(cropped_image_female)
            else:
                writer_small_female.writeFrame(filler_frame)

            if cropped_image_male.shape == (100, 100) and cropped_image_female.shape == (100, 100):
                writer_small.writeFrame(np.concatenate((cropped_image_male, cropped_image_female), axis=1))
            else:
                writer_small.writeFrame(np.concatenate((filler_frame, filler_frame), axis=1))
            # position = pos
            last_ori = ori
            last_position = position
            frame_number = frame_number + 1
            video_frame = video_frame + 1

    # camera.stopCapture()
    # camera.disconnect()
    # camera.Close()
    cv2.destroyAllWindows()
    # win.close()
    writer_small.close()
    writer_small_female.close()
    writer_small_male.close()
    csv_file.close()

    # print('here')

    json.dump(total_data, f_json)
    f_json.close()
    f_pickle.close()

    if os.path.isfile(
            files_path_json.strip(
                '.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
        os.remove(files_path_json.strip('.json') + 'copy' + '.json')
    exit()


