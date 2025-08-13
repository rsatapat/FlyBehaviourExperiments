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
#num_bars = [4,6,10]
num_bars = [6]
#freq = [1, 2, 5, 10, 15, 20]
freq = [2,5,10]
bg = int(exp_info[4])
fg = int(exp_info[5])
#size_stim = [0.25, 0.5, 0.75, 1.0]
size_stim = [1]
bright_bars = np.linspace(0.0, 1.0, 11)
dark_bars = np.linspace(0.0, 1.0, 11)
date_time = datetime.datetime.now()
stimulus = 'pinwheel'

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
camera = image_methods.initialise_camera("ptgrey")

##psychopy stimulus window initialisation
win = psychopy.visual.Window(
    size=[342, 684],
    pos=[1998, -20],
    color=(0, 0, 0),
    fullscr=False,
    waitBlanking=True
)

win.recordFrameIntervals = True
win.refreshThreshold = 1 / 60 + 0.004

##first image taken for pre-processing
image, timestamp = image_methods.grab_image(camera, 'ptgrey')
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

## this is where the experiment starts
time_exp_start = time.clock()

trials = 0
video_frame = 0
while (trials < 30):
    a = [1, -1]
    direction = random.choice(a)
    bars = random.choice([6])
    Hz = random.choice([5])
    stim_size = random.choice([0.5,1])
    contrast = random.choice([100])
    angle = (360 * Hz) / (framerate * bars)
    print(bars, angle)
    # closedloop_gain = random.choice([0,0.25,0.5,0.75,1])
    #closedloop_gain = random.choice([-1,-0.5,0,0.25,0.5,0.75,1,2])
    closedloop_gain = 0

    stimulus_image = stimulus + '_' + str(contrast) + '_' + str(bars) + '.png'
    stimulus_image = r'{}\{}'.format('D:\Roshan\Project\Python_codes\Stimulus\Generate stimulus', stimulus_image)

    wheel_stim = psychopy.visual.ImageStim(
        win=win,
        image=stimulus_image,
        mask='circle',
        pos=(0, 0),
        ori=0,  # 0 is vertical,positive values are rotated clockwise
        size=1,
    )
    wheel_stim.autoDraw = False

    dot_stim = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1 / 512,
        pos=(0, 0)
    )
    dot_stim.autoDraw = False

    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)

    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pos, ellipse, cnt = fly_track.fly_court_pos(contours)
    wheel_stim.ori = 180 - ellipse[2]
    # main loop
    # to capture and save images
    # to find the location of fly
    # to project stimulus to the required point
    frame_number = 0
    cv2.destroyAllWindows()
    position = (0, 0)
    filler_frame = np.zeros((60, 60))
    time_stim_start = time.clock()
    fly_ori = 0
    fly_ori_last = ellipse[2]
    fly_frame_ori_last = ellipse[2]
    while (True):  # this unit(of experiment) is repeated #
                    # stimulus is not re-defined inside this loop
                    # only parameters are changed ##
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')

        cv_image = ROI(image, loc, size)

        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pos, ellipse, cnt = fly_track.fly_court_pos(contours)



        if len(pos) != 1:
            frames_lost += 1  # counts number of frames in which the location of fly could not be determined
            print('fly lost')
            pos.append(position)
            if frames_lost > 150:  # if the number of lost frames is more than 15, abort experiment
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

        x = (pos[0][0] - 512) / w_ratio + w_rem
        y = (pos[0][1] - 512) / h_ratio + h_rem
        wheel_stim.pos = [[x, y]]
        dot_stim.pos = [[x, y]]



        if frame_number < 13 * framerate:
            dir = 0
        elif frame_number >= 13 * framerate and frame_number < 14 * framerate:
            dot_stim.fillColor = (-1, -1, -1)
            dot_stim.radius = dot_stim.radius + (2 / 512)

            #print('drawing dot')
            dir = 0
        elif frame_number >= 14 * framerate and frame_number < 15 * framerate:
            dot_stim.radius = 1/512
            dir = 0
        elif frame_number >= 15 * framerate and frame_number < 20 * framerate:
            dir = direction

        elif frame_number >= 20 * framerate and frame_number < 21 * framerate:
            dot_stim.fillColor = (-1, -1, -1)
            dot_stim.radius = dot_stim.radius + (2 / 512)

            #print('drawing dot')
            dir = direction
        elif frame_number >= 21 * framerate and frame_number < 30 * framerate:
            dot_stim.radius = 1 / 512
            dir = direction



        elif frame_number >= 30 * framerate and frame_number < 43 * framerate:
            dir = 0

        elif frame_number >= 43 * framerate and frame_number < 44 * framerate:
            dot_stim.fillColor = (-1, -1, -1)
            dot_stim.radius = dot_stim.radius + (2 / 512)

            # print('drawing dot')
            dir = 0
        elif frame_number >= 44 * framerate and frame_number < 45 * framerate:
            dot_stim.radius = 1 / 512
            dir = 0
        elif frame_number >= 45 * framerate and frame_number < 50 * framerate:
            dir = (-1) * direction

        elif frame_number >= 50 * framerate and frame_number < 51 * framerate:
            dot_stim.fillColor = (-1, -1, -1)
            dot_stim.radius = dot_stim.radius + (2 / 512)

            #print('drawing dot')
            dir = (-1) * direction
        elif frame_number >= 51 * framerate and frame_number < 60 * framerate:
            dot_stim.radius = 1 / 512
            dir = (-1) * direction

        elif frame_number >= 60 * framerate:
            stimulus_inst = objects.stim_inst([stimulus, stimulus_image], direction, bars, Hz,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_stim_start,
                                          frame_number, video_frame, contrast, stim_size)
            stimulus_inst.stim_attributes = [wheel_stim.mask,closedloop_gain]
            exp_number = len(total_data['exp_params'])
            total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
            pickle.dump(stimulus_inst.__dict__, f_pickle)
            break
        wheel_stim.ori = (wheel_stim.ori + dir * angle - closedloop_gain*fly_turn) % 360
        wheel_stim.draw()
        dot_stim.draw()

        win.flip()  # slowest part of the algorithm,takes ~10ms

        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp, dir])
        cropped_image = cv_image[int(pos[0][1]) - 30:int(pos[0][1]) + 30, int(pos[0][0]) - 30:int(pos[0][0]) + 30]
        if cropped_image.shape == (60, 60):
            writer_small.writeFrame(cropped_image)
        else:
            print("cropping failure...")
            writer_small.writeFrame(filler_frame)
        # cv2.imshow('frame', cv_image)
        position = pos[0]
        fly_ori_last = fly_ori
        fly_frame_ori_last = fly_frame_ori
        frame_number = frame_number + 1
        video_frame = video_frame + 1
    time.sleep(10)
    trials += 1

camera.stopCapture()
camera.disconnect()
cv2.destroyAllWindows()
writer_small.close()
csv_file.close()

json.dump(total_data, f_json)
f_json.close()
f_pickle.close()

if os.path.isfile(
        files_path_json.strip('.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
    os.remove(files_path_json.strip('.json') + 'copy' + '.json')
