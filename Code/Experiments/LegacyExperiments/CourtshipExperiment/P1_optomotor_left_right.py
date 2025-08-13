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
import serial as ser
import random
import string
def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
exp_info = info()
# exp_info = [get_random_string(5), 'M', '0', '1', '1', '-1', '_']
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
stimulus = 'pinwheel_and_female_dot'

Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
# Dir = r'D:\Roshan'
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
writer_csv.writerow(['pos_x', 'pos_y', 'ori', 'timestamp','female_angle','opto_stim','optomotor','pinwheel_angle'])
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
camera = image_methods.initialise_camera("basler")

##psychopy stimulus window initialisation
win = psychopy.visual.Window(
    size=[342, 684],
    pos=[1920+120, 15],
    color=(1, 1, 1),
    fullscr=False,
    waitBlanking=True,
    gammaErrorPolicy='ignore',
    winType='glfw'
    # winType='pygame'
)
win.recordFrameIntervals = True
win.refreshThreshold = 1 / 60 + 0.004

# ##########################
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

## this is where the experiment starts
time_exp_start = time.clock()
dot_stim = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1/15,
        pos=(0, 0),
        lineColor=1,
        fillColor=1
)
dot_stim.autoDraw = False

dot_stim2 = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1/15,
        pos=(0, 0),
        lineColor=1,
        fillColor=1
)
dot_stim2.autoDraw = False

ard_ser = ser.Serial('COM3', 57600)
time.sleep(2)
ard_ser.write(b'H')

position = (0, 0)
filler_frame = np.zeros((120, 120))

image, timestamp = image_methods.grab_image(camera, 'basler')
cv_image = ROI(image, loc, size)
ret, diff_img = cv2.threshold(cv_image, 40, 255, cv2.THRESH_BINARY)
this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
pos, ellipse, cnt = fly_track.fly_court_pos(contours)
fly_ori_last = ellipse[2]
fly_frame_ori_last = ellipse[2]
##########################
offset = 0
while(True):
    image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=100, max_size=6000)

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
    cv2.imshow('frame2', cropped_image)
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

trials = 0
video_frame = 0
# fly_ori_last = ellipse[2]
# fly_frame_ori_last = ellipse[2]
frame_lost = 0
FREQS = [20, 25, 40, 50, 100]
frames_lost = 0
while (trials < 90):
    # main loop
    # to capture and save images
    # to find the location of fly
    # to project stimulus to the required point
    ######################################
    bars = random.choice([6])
    contrast = random.choice([10, 25])
    stimulus_image_pinwheel = 'pinwheel' + '_' + str(contrast) + '_' + str(bars) + '.png'
    stimulus_image_pinwheel = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image_pinwheel)
    stim_size = size_stim
    wheel_stim = psychopy.visual.ImageStim(
        win=win,
        image=stimulus_image_pinwheel,
        mask='circle',
        pos=(0, 0),
        ori=0,  # 0 is vertical,positive values are rotated clockwise
        size=0.8,
    )
    wheel_stim.autoDraw = False
    ######################################
    closedloop_gain = 0
    frame_number = 0
    time_stim_start = time.clock()
    fly_ori = 0
    frame_lost = 0
    dist_female = 250
    speed_of_female = 2
    stimulation_time = 10   #in seconds
    on_time = 10
    frequency = FREQS[3]
    off_time = int((1000/frequency) - on_time)
    repetitions = int(stimulation_time*1000/(on_time+off_time))
    input_str = str(on_time)+','+str(off_time)+','+str(repetitions)
    angle = -100
    x, y = -1, -1
    stim = 0
    direction = random.choice([-1,1])
    Hz = random.choice([1,4,8])
    # dot_stim2.fillColor = random.choice([-1, -0.2, -0.5, -0.8])
    while (True):
        image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=100, max_size = 4000)
        ################

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

        if frame_number == 0:
            dir = 0
            optomotor_stim = 0
            camera.TimestampLatch()
            t1 = camera.TimestampLatchValue.GetValue()
            if 15>trials>=10:
                ard_ser.write(str.encode(input_str))
                stim = 1
        elif frame_number >= 0 * framerate and frame_number < 6 * framerate:
            stim = 1
            # angle = 45-abs(90 - (speed_of_female*frame_number % 180))
            angle = frame_number * 2 * speed_of_female % 360
            angle = 45 * (np.sin((angle / 180) * np.pi))
            angle = angle * Hz
            optomotor_stim = 0
            dir = 0
            x = (pos[0][0] - 512) / w_ratio + w_rem
            y = (pos[0][1] - 512) / h_ratio + h_rem
            wheel_stim.pos = [[x, y]]
            x_dot, y_dot = int(dist_female * np.cos((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][0]), int(
                dist_female * np.sin((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][1])
            x_dot = (x_dot - 512) / w_ratio + w_rem
            y_dot = (y_dot - 512) / h_ratio + h_rem
            dot_stim.pos = [[x_dot, y_dot]]
            # wheel_stim.ori = (wheel_stim.ori + dir * angle_opto) % 360
            wheel_stim.draw()
            dot_stim.draw()
        elif frame_number > 6 * framerate and frame_number < 12 * framerate:
            stim = 0
            # angle = 45-abs(90 - (speed_of_female*frame_number % 180))
            angle = frame_number * 2 * speed_of_female % 360
            angle = 45 * (np.sin((angle / 180) * np.pi))
            angle = angle * Hz
            dir = direction
            optomotor_stim = dir
            x = (pos[0][0] - 512) / w_ratio + w_rem
            y = (pos[0][1] - 512) / h_ratio + h_rem
            wheel_stim.pos = [[x, y]]
            x_dot, y_dot = int(dist_female * np.cos((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][0]), int(
                dist_female * np.sin((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][1])
            x_dot = (x_dot - 512) / w_ratio + w_rem
            y_dot = (y_dot - 512) / h_ratio + h_rem
            dot_stim.pos = [[x_dot, y_dot]]
            wheel_stim.ori = angle * -1
            # wheel_stim.ori = wheel_stim.ori + Hz * 5 * random.random() * random.choice([1, -1])
            wheel_stim.draw()
            dot_stim.draw()
        elif frame_number >= 12 * framerate:
            print('Stimulation off...')
            stimulus_inst = objects.stim_inst([stimulus, float(dot_stim.radius), list(dot_stim.fillColor), bars, contrast, Hz],on_time, off_time, frequency,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_stim_start,
                                          frame_number, video_frame, speed_of_female, dist_female)
            # stimulus_inst.stim_attributes = ['optogenetic']
            exp_number = len(total_data['exp_params'])
            total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
            pickle.dump(stimulus_inst.__dict__, f_pickle)
            break
        win.flip()

        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp-t1,angle, stim, dir, -1*wheel_stim.ori])
        cropped_image = cv_image[int(pos[0][1]) - 60:int(pos[0][1]) + 60, int(pos[0][0]) - 60:int(pos[0][0]) + 60]

        if cropped_image.shape == (120, 120):
            writer_small.writeFrame(cropped_image)
        else:
            print("cropping failure...")
            cropped_image = filler_frame
            writer_small.writeFrame(filler_frame)
        cropped_image = cv2.line(cropped_image, (60,60),\
            (int(100*np.cos(((fly_frame_ori/180)*np.pi)+np.pi/2)+60), int(100*np.sin(((fly_frame_ori/180)*np.pi)+np.pi/2)+60)), (255,255,255), 2)
        cv2.imshow('frame2', cropped_image)
        input = cv2.waitKey(1)
        if input == ord('m'):
            print('Is this the right direction?')
            fly_frame_ori = fly_frame_ori + 180
        else:
            pass

        cv2.imshow('frame2', cropped_image)
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

win.saveFrameIntervals(clear = True)
file_name_time = Dir_new + '_time.txt'
files_path_time = r'{}\{}'.format(Dir_new, file_name_time)
win.saveFrameIntervals(fileName = files_path_time, clear=True)

if os.path.isfile(
        files_path_json.strip('.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
    os.remove(files_path_json.strip('.json') + 'copy' + '.json')
