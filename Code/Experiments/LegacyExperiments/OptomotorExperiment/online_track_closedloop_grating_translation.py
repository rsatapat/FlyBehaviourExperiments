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


exp_info = info()

# fly_parameters
strain = exp_info[0]  # fly strain
serial = exp_info[3]  # fly_id
sex = exp_info[1]
age = int(exp_info[2])
fly = objects.fly(strain, serial, sex, age)  ## fly object defined here
if exp_info[6] == '_':
    remark = ''
else:
    remark = '_' + exp_info[6]

# video parameters
framerate = 60
resolution = (720, 720)
vid = objects.video(resolution, framerate)  ## vid object defined here

# experiment parameters
bg = int(exp_info[4])
fg = int(exp_info[5])
date_time = datetime.datetime.now()
stimulus = 'sine_pinwheel'

Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
if os.path.isdir(Dir) == False:  ## checks if this folder exists
    print('Folder name incorrect')
    exit()

date_string = ''
for i in [date_time.year, date_time.month, date_time.day]:
    date_string = date_string + str(i)

Dir_name = strain + '_' + serial
Dir_new = r'{}\{}'.format(Dir, Dir_name)

if os.path.isdir(Dir_new) == False:  ## checking to see of the fly already has a folder or not
    os.mkdir(Dir_new)  ## if no such folder exists, make one

file_name_json = Dir_name + '.json'
file_name_pickle = Dir_name + '.p'

files_path_json = r'{}\{}'.format(Dir_new, file_name_json)
files_path_pickle = r'{}\{}'.format(Dir_new, file_name_pickle)
f_pickle = open(files_path_pickle, 'ab')

## checking to see if a json file with the same name exists
exists = os.path.isfile(files_path_json)
if exists:
    shutil.copyfile(files_path_json, files_path_json.strip('.json') + 'copy' + '.json')
    f_json_read = open(files_path_json, 'r')
    existing_data = json.load(f_json_read)
    existing_data_copy = copy.deepcopy(existing_data)  ## create a deep copy of the existing_data
    del existing_data_copy['exp_params']
    code = misc.compare_dicts(fly.__dict__,
                              existing_data_copy)  ## change code here to include only the fly data and not any other data
    if code == -1:
        print('There is something wrong with the data, please check manually')
        exit()
    elif code == -2:
        print('Incorrect folder selected, fly data has changed')
        exit()
    elif code == 1 or code == 2:
        total_data = {**fly.__dict__, **{'exp_params': existing_data['exp_params']}}  # fly_data added to the dictionary
    f_json_read.close()
else:
    total_data = {**fly.__dict__, **{'exp_params': []}}  # fly_data added to the dictionary

f_json = open(files_path_json,
              'w')  ## open the json file ## important : data is not appended but re-written at the end of the experiment
pickle.dump(fly.__dict__,
            f_pickle)  ## start writing to the pickle file, it acts as a temporary storage, in case of ant failure in writing to the json file

filename = strain + '_' + serial + '_' + date_string + remark
filename = r'{}\{}'.format(Dir_new, filename)
csv_file = filename + '.csv'
if os.path.isfile(csv_file):
    print('This file already exists, please add a remark or identifier to uniquely identify this video')
    reply = message_box.data_change_error('Warning : File already exists', 'Proceed?')
    if reply == 1:
        filename = misc.change_file_name(filename)
        csv_file = filename + '.csv'
    else:
        exit()
csv_file = open(csv_file, 'w')
writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer_csv.writerow(['pos_x', 'pos_y', 'ori', 'timestamp', 'direction'])
writer_small = sk.FFmpegWriter(filename + '.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264',
                                                                                      '-crf': '0',
                                                                                      '-preset': 'slow'})

## experiment object defined here
experiment = objects.exp([date_time.year, date_time.month, date_time.day],
                         [date_time.hour, date_time.minute, date_time.second, date_time.microsecond], bg, fg, filename)
total_data['exp_params'].append(
    {**vid.__dict__, **experiment.__dict__, **{'stim_params': []}})  ## experiment data added to the dictionary
pickle.dump({**vid.__dict__, **experiment.__dict__}, f_pickle)


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
# camera = image_methods.initialise_camera("ptgrey")
camera = image_methods.initialise_camera("basler")

# ##psychopy stimulus window initialisation
# win = psychopy.visual.Window(
#     size=[342, 684],
#     pos=[2002, -10],
#     color=(0, 0, 0),
#     fullscr=False,
#     waitBlanking=True
# )

#psychopy stimulus window initialisation
win = psychopy.visual.Window(
    size=[342, 684],
    # pos = [1998,-20],
    pos=[1920+120, 15],
    color = (0,0,0),
    fullscr=False,
    waitBlanking = True
   # winType='pyglet'
)
win.recordFrameIntervals = True
win.refreshThreshold = 1 / 60 + 0.004
win.saveFrameIntervals(fileName=r'{}/{}'.format(Dir, 'file.log'))

##########################
# win = psychopy.visual.Window(
#     size=[480, 480],
#     # pos = [1998,-20],
#     monitor = 0,
#     pos=[1920+240, -35],
#     color = (0,0,0),
#     fullscr=False,
#     waitBlanking = True
#    # winType='pyglet'
# )
# win.recordFrameIntervals = True
# win.refreshThreshold = 1 / 120 + 0.001
# print(Dir)
# win.saveFrameIntervals(fileName=r'{}/{}'.format(Dir, 'file.log'))
##########################

##first image taken for pre-processing
# image, timestamp = image_methods.grab_image(camera, 'ptgrey')
image, timestamp = image_methods.grab_image(camera, 'basler')
height, width = image.shape

##user finds the arena, by selecting ROI
arena = cv2.selectROI(image, False)
loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
size = arena[2] // 2

cv_image = ROI(image, loc, size)

w_ratio = 512
w_rem = 0
h_ratio = 512
h_rem = 0

offset = 0
position = (0, 0)
filler_frame = np.zeros((120, 120))
image, timestamp = image_methods.grab_image(camera, 'basler')
cv_image = ROI(image, loc, size)
ret, diff_img = cv2.threshold(cv_image, 40, 255, cv2.THRESH_BINARY)
this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
pos, ellipse, cnt = fly_track.fly_court_pos(contours)
fly_ori_last = ellipse[2]
fly_frame_ori_last = ellipse[2]
while(True):
    image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 50, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=50, max_size=8000)

    if len(pos) != 1:
        frames_lost += 1  # counts number of frames in which the location of fly could not be determined
        print('fly lost')
        pos.append(position)
        if frames_lost > 300:  # if the number of lost frames is more than 15, abort experiment
            print('Too many frames are being lost')
            break
    else:
        frames_lost = 0
        fly_ori = ellipse[2]
        fly_turn = fly_ori - fly_ori_last
        if fly_turn > 90:
            fly_turn = -(180 - fly_turn)
        elif fly_turn < -90:
            fly_turn = 180 + fly_turn
        fly_frame_ori = fly_frame_ori_last + fly_turn
        # print(fly_ori, fly_ori_last, time.clock())

    cropped_image = cv_image[int(pos[0][1]) - 60:int(pos[0][1]) + 60, int(pos[0][0]) - 60:int(pos[0][0]) + 60]
    if cropped_image.shape == (120, 120):
        pass
    else:
        print("cropping failure...")
        cropped_image = filler_frame
    cropped_image = cv2.line(cropped_image, (60, 60), (int(100 * np.cos((((fly_frame_ori%360)/180) * np.pi) + np.pi / 2) + 60),
                    int(100 * np.sin((((fly_frame_ori%360)/180) * np.pi) + np.pi / 2) + 60)),(255, 255, 255), 2)
    cv2.imshow('frame1', cropped_image)
    input = cv2.waitKey(4)
    if input == ord('m'):
        print('Is this the right direction?')
        fly_frame_ori = fly_frame_ori + 180
    elif input == ord('q'):
        print('Starting Experiment')
        break
    else:
        pass
    fly_ori_last = fly_ori
    fly_frame_ori_last = fly_frame_ori
##################################

## this is where the experiment starts
time_exp_start = time.clock()
trials = 0
video_frame = 0
print('video_frame')
##########################
trials = 0
while (trials < 40):
    bars = random.choice([4,6])
    contrast = random.choice([100])
    # Hz = random.choice([2, 5, 10, 15, 20, 25])
    direction = random.choice([1, -1])
    # direction = 1
    closedloop_gain = random.choice([0])
    Hz = random.choice([0.5, 1, 2, 4, 10])
    ########################################
    grating_stim = psychopy.visual.GratingStim(win, tex='sqr', mask='none', units='norm', pos=(0.0, 0.0), size=4, sf=8, ori=0.0,
                            phase=0,  texRes=128, color=(1.0, 1.0, 1.0), colorSpace='rgb',
                            contrast=1.0, depth=0, interpolate=False, blendmode='avg',
                            name=None, autoLog=None, autoDraw=False, maskParams=None)

    # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_track.fly_court_pos(contours)
    grating_stim.ori = 180 - ellipse[2]
    # main loop
    # to capture and save images
    # to find the location of fly
    # to project stimulus to the required point
    frame_number = 0
    cv2.destroyAllWindows()
    position = (0, 0)
    filler_frame = np.zeros((120, 120))
    time_stim_start = time.clock()
    # fly_ori = 0
    # fly_ori_last = ellipse[2]
    # fly_frame_ori_last = ellipse[2]
    t1=0
    real_angle = 0
    while (True):
        # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=100, max_size = 4000)

        fly_ori = ellipse[2]
        fly_turn = fly_ori - fly_ori_last

        if fly_turn > 90:
            fly_turn = -(180 - fly_turn)
        elif fly_turn < -90:
            fly_turn = 180 + fly_turn

        fly_frame_ori = fly_frame_ori_last + fly_turn
        # print('here')
        # print(fly_frame_ori%360)
        ################################################
        ################################################
        #### head abodomen direction
        # M = cv2.moments(cnt[0])
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        # cropped_image = cv_image[int(pos[0][1]) - 60:int(pos[0][1]) + 60, int(pos[0][0]) - 60:int(pos[0][0]) + 60]
        # ret, diff_img = cv2.threshold(cropped_image, 60, 255, cv2.THRESH_BINARY)
        # ret, diff_img1 = cv2.threshold(cropped_image, 130, 255, cv2.THRESH_BINARY)
        # wing_img = diff_img - diff_img1
        # ## erosion to get rid of the pixels that appear around the body
        # kernel = np.ones((2, 2), np.uint8)
        # erosion = cv2.erode(wing_img, kernel, iterations=1)
        # cv2.imshow('erosion', erosion)
        # input = cv2.waitKey(1)
        # ## contour of the remaining (hopefully only wing)
        # _, contours_eroded, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #
        # ## try to find wings here
        # area_max = 50
        # wing = 0
        # cx2 = 0
        # cy2 = 0
        # for i in range(len(contours_eroded)):
        #     Area = cv2.contourArea(contours_eroded[i])
        #     # print(Area)
        #     if Area > area_max:
        #         # print(Area)
        #         # area_max = Area
        #         wing = wing+1
        #         M = cv2.moments(contours_eroded[i])
        #         ## new image created with only the male fly
        #         image_test = np.zeros((100, 100), np.uint8) ## shows wings
        #         cv2.drawContours(image_test, contours_eroded, i, 255, -1)
        #         image_test = cv2.cvtColor(image_test, cv2.COLOR_GRAY2BGR)
        #         cx2 = cx2+int(M['m10'] / M['m00'])
        #         cy2 = cy2+int(M['m01'] / M['m00'])
        #         # cv2.imshow('frame1', image_test)
        #         # cv2.waitKey(0)
        #
        # ## centroid of the wings
        # if wing == 0:
        #     cx2=cx
        #     cy=0
        # else:
        #     cx2 = cx2/wing
        #     cy2 = cy2/wing
        # # print(cy, cy2, cx, cx2)
        # body_angle = (((np.arctan2(cy - cy2, cx2 - cx) / np.pi) * 180) + 90) % 360
        # print(cx2, cy2)
        # print(body_angle%360)
        # fly_ori_180 = fly_ori + 180
        # diff = abs(body_angle - fly_ori)
        # diff2 = abs(body_angle - fly_ori_180)
        # print('difference {}/{}'.format(diff, diff2))
        # if diff < 80 or diff > 300:
        #     print('1')
        #     real_angle = fly_ori + 180
        #     # real_angle = fly_frame_ori
        # elif diff2 < 80 or diff2 > 300:
        #     print('2')
        #     real_angle = fly_ori
        #     # real_angle = fly_frame_ori + 180
        # else:
        #     print('head-tail {},{}'.format(diff, diff2))
        #     # fly_frame_ori_180 = fly_frame_ori + 180
        #     diff = abs(real_angle - fly_ori)
        #     diff2 = abs(real_angle - fly_ori_180)
        #     if diff < 60 or diff > 300:
        #         real_angle = fly_ori
        #         # real_angle = fly_frame_ori + 180
        #     elif diff2 < 60 or diff2 > 300:
        #         real_angle = fly_ori + 180
        #         # real_angle = fly_frame_ori
        #     else:
        #         real_angle = fly_ori
        ############################################
        # diff = (body_angle - (fly_frame_ori% 360))%360
        # if diff < 80 or diff > 300:
        #     fly_ori = fly_ori + 180
        # else:
        #     pass
        # fly_turn = fly_ori - fly_ori_last
        # print(fly_turn)
        # # if fly_turn > 90:
        # #     fly_turn = -(180 - fly_turn)
        # # elif fly_turn < -90:
        # #     fly_turn = 180 + fly_turn
        #
        # fly_frame_ori = fly_frame_ori_last + fly_turn
        ################################################
        ################################################
        if len(pos) != 1:
            frames_lost += 1  # counts number of frames in which the location of fly could not be determined
            print('fly lost')
            pos.append(position)
            if frames_lost > 1000:  # if the number of lost frames is more than 15, abort experiment
                print('Too many frames are being lost')
                break
        else:
            frames_lost = 0

        x = (pos[0][0] - 512) / w_ratio + w_rem
        y = (pos[0][1] - 512) / h_ratio + h_rem
        grating_stim.pos = [[0, 0]]

        if frame_number == 0:
            camera.TimestampLatch()
            t1 = camera.TimestampLatchValue.GetValue()
            dir = 0
        elif frame_number < 5 * framerate:
            dir = 0
        elif frame_number >= 5 * framerate and frame_number < 10 * framerate:
            grating_stim.ori = fly_frame_ori
            dir = direction ## from front to back
        elif frame_number >= 10 * framerate and frame_number < 15 * framerate:
            dir = 0
        elif frame_number >= 15 * framerate and frame_number < 20 * framerate:
            grating_stim.ori = fly_frame_ori
            dir = (-1) * direction ## from back to front
        elif frame_number >= 20 * framerate:
            stimulus_inst = objects.stim_inst([stimulus], direction, bars, Hz,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_stim_start,
                                          frame_number, video_frame, contrast, 0)
            stimulus_inst.stim_attributes = [grating_stim.mask, closedloop_gain]
            exp_number = len(total_data['exp_params'])
            total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
            pickle.dump(stimulus_inst.__dict__, f_pickle)
            break
        grating_stim.phase = grating_stim.phase + Hz * (1/60) * dir
        grating_stim.draw()
        win.flip()  # slowest part of the algorithm,takes ~10ms
        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp  - t1, dir])
        cropped_image = cv_image[int(pos[0][1]) - 60:int(pos[0][1]) + 60, int(pos[0][0]) - 60:int(pos[0][0]) + 60]
        if cropped_image.shape == (120, 120):
            pass
        else:
            print("cropping failure...")
            cropped_image = filler_frame
        writer_small.writeFrame(cropped_image)
        cropped_image = cv2.line(cropped_image, (60,60),\
            (int(100*np.cos(((fly_frame_ori/180)*np.pi)+np.pi/2)+60), int(100*np.sin(((fly_frame_ori/180)*np.pi)+np.pi/2)+60)), (255,255,255), 2)
        cv2.imshow('frame3', cropped_image)
        input = cv2.waitKey(1)
        if input == ord('m'):
            print('Is this the right direction?')
            fly_frame_ori = fly_frame_ori + 180
        else:
            pass

        position = pos[0]
        fly_ori_last = fly_ori
        fly_frame_ori_last = fly_frame_ori
        frame_number = frame_number + 1
        video_frame = video_frame + 1
    trials += 1

# camera.stopCapture()
# camera.disconnect()
cv2.destroyAllWindows()
writer_small.close()
csv_file.close()

json.dump(total_data, f_json)
f_json.close()
f_pickle.close()

if os.path.isfile(
        files_path_json.strip('.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
    os.remove(files_path_json.strip('.json') + 'copy' + '.json')

json.dump(total_data, f_json)
f_json.close()

exit()
