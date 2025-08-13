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
framerate = 220
resolution = (512, 512)
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
writer_csv.writerow(['pos_x', 'pos_y', 'fly_frame_ori', 'timestamp', 'stim'])
writer_small = sk.FFmpegWriter(filename + '.avi', outputdict={'-vcodec': 'libx264',
                                                                '-crf': '0',
                                                                '-preset': 'medium'})
## inputdict={'-r': str(framerate)},
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

#psychopy stimulus window initialisation
# win = psychopy.visual.Window(
#     pos=[0, 0],
#     screen = 1,
#     color = (-1,-1,-1),
#     fullscr=True,
#     waitBlanking=True
# )

# win.recordFrameIntervals = True
# win.refreshThreshold = 1 / 60 + 0.004

##first image taken for pre-processing
# image, timestamp = image_methods.grab_image(camera, 'ptgrey')
image, timestamp = image_methods.grab_image(camera, 'basler')
height, width = image.shape

#user finds the arena, by selecting ROI
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
# image, timestamp = image_methods.grab_image(camera, 'ptgrey')
image, timestamp = image_methods.grab_image(camera, 'basler')
# cv_image = ROI(image, loc, size)
cv_image = image
ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)

this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

pos, ellipse, cnt = fly_track.fly_court_pos(contours)
position = (0, 0)
filler_frame = np.zeros((120, 120))
time_stim_start = time.clock()
fly_ori = 0
fly_ori_last = ellipse[2]
fly_frame_ori_last = ellipse[2]
t = time.clock()
pos = [[0,0]]

ard_ser = ser.Serial('COM3', 57600)
time.sleep(2)
print('Established connection with arduino')
# FREQS = [20, 25, 40, 50, 100]
FREQS = [50, 100]
while (trials < 60):
    stim = 0
    intensity = 1
    frame_number = -5000
    time_stim_start = time.clock()
    # color = random.choice([(1, 1, -1), (1, 1, -1)])
    see_fly = 0
    stimulation_time = 2   #in seconds
    on_time = 10
    frequency = random.choice(FREQS)
    off_time = int((1000/frequency) - on_time)
    repetitions = int(stimulation_time*1000/(on_time+off_time))
    input_str = str(on_time)+','+str(off_time)+','+str(repetitions)
    # t = time.time()
    image_list = []
    while(True):
        # t = time.clock()
        # image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        # cv_image = image
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=200, max_size=7000)

        if len(pos) != 1:
            pos = []
            pos.append(position)
            fly_frame_ori = fly_ori
            pass
        else:
            position = pos[0]
            fly_ori = ellipse[2]
            fly_turn = fly_ori - fly_ori_last

            if fly_turn > 90:
                fly_turn = -(180 - fly_turn)
            elif fly_turn < -90:
                fly_turn = 180 + fly_turn

            fly_frame_ori = fly_frame_ori_last + fly_turn

            fly_ori_last = fly_ori
            fly_frame_ori_last = fly_frame_ori
            if see_fly == 0:
                if pos[0][0] > 320 and pos[0][0] < 1024-320 and pos[0][1] > 320 and pos[0][1] < 1024-320:
                    see_fly = 1
                    frame_number = 0
                    print('fly_detected')
            else:
                pass
        # x = (pos[0][0] - 512) / w_ratio + w_rem
        # y = (pos[0][1] - 512) / h_ratio + h_rem
        if see_fly == 1:
            video_frame = video_frame + 1
            if frame_number == 0:
                camera.TimestampLatch()
                t1 = camera.TimestampLatchValue.GetValue()
            elif frame_number == 1 * framerate:
                print('Stimulation on..')
                stim = random.choice([0,1,1,1,1,1,1])
                if stim == 1:
                    print(input_str)
                    print(60 - trials)
                    # ard_ser.write(b'L')
                    ard_ser.write(str.encode(input_str))
                else:
                    # ard_ser.write(b'H')
                    pass
            elif frame_number > 1 * framerate and frame_number < (1+stimulation_time) * framerate:
                pass
            elif frame_number == (1+stimulation_time) * framerate:
                stim = 0
                # ard_ser.write(b'H')
            elif frame_number > (stimulation_time+1+1.5) * framerate:
                frame_number = -5000
                see_fly=0
                stim = 0
                print('Stimulation off...')
                stimulus_inst = objects.stim_inst([0], on_time, frequency, stimulation_time,
                                              time.clock() - time_exp_start,
                                              time.clock() - time_stim_start,
                                              frame_number, video_frame, off_time, 0)
                stimulus_inst.stim_attributes = ['optogenetic', stimulation_time]
                # stimulus_inst.stim_attributes = ['optogenetic']
                exp_number = len(total_data['exp_params'])
                total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
                pickle.dump(stimulus_inst.__dict__, f_pickle)
                time.sleep(5)
                break
            writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp - t1, stim])

            cropped_image = cv_image[int(pos[0][1]) - 60:int(pos[0][1]) + 60, int(pos[0][0]) - 60:int(pos[0][0]) + 60]
            if cropped_image.shape == (120, 120):
                writer_small.writeFrame(cropped_image)
            else:
                # print("cropping failure...")
                cropped_image = filler_frame
                writer_small.writeFrame(filler_frame)
            cropped_image = cv2.line(cropped_image, (60, 60), \
                                     (int(100 * np.cos(((fly_frame_ori / 180) * np.pi) + np.pi / 2) + 60), int(100 * np.sin(((fly_frame_ori / 180) * np.pi) + np.pi / 2) + 60)),
                                     (255, 255, 255), 2)
            cv2.imshow('frame', cropped_image)
            cv2.waitKey(1)
            if input == ord('m'):
                print('Is this the right direction?')
                fly_frame_ori = fly_frame_ori + 180
            else:
                pass

        frame_number = frame_number + 1
        # print(time.clock()-t)
    trials = trials + 1

# camera.stopCapture()
# camera.disconnect()
cv2.destroyAllWindows()
writer_small.close()
csv_file.close()

json.dump(total_data, f_json)
f_json.close()
f_pickle.close()

def convert_video(Dir, src, target):
    import subprocess
    src = r'{}/{}'.format(Dir, src)
    target = r'{}/{}'.format(Dir, src+target)
    subprocess.call('ffmpeg -i src -pix_fmt nv12 -f avi -vcodec rawvideo target')

if os.path.isfile(
        files_path_json.strip('.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
    os.remove(files_path_json.strip('.json') + 'copy' + '.json')
