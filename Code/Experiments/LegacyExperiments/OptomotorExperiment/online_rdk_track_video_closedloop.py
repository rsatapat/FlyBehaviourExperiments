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
stimulus_duration = int(exp_info[7])
# video parameters
framerate = 60
resolution = (720, 720)
vid = objects.video(resolution, framerate)  ## vid object defined here

# experiment parameters
num_bars = [4, 6, 12]


bg = int(exp_info[4])
fg = int(exp_info[5])
freq = [2,5,10]
size_stim = [0.25, 0.5, 0.75, 1.0]
size_stim = [1.0]
bright_bars = np.linspace(0.0, 1.0, 11)
dark_bars = np.linspace(0.0, 1.0, 11)
date_time = datetime.datetime.now()
stimulus = 'RDK'

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
# camera = image_methods.initialise_camera("basler")

win = misc.initialise_projector_window(Dir, pos=[0, 0], size=[342//2, 684//2])

##first image taken for pre-processing
image, timestamp = image_methods.grab_image(camera, 'ptgrey')
# image, timestamp = image_methods.grab_image(camera, 'basler')
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
while (trials < 160):
    ##parameters
    N = random.choice([80, 160])
    coherence = random.choice([0.5, 0.6, 0.8, 1])
    n1 = int(N * coherence)
    n2 = int(N * (1 - coherence))
    stim_size = random.choice([0.1])
    speed = random.choice([10])
    angle = speed
    direction = random.choice([1, -1])
    contrast = random.choice([-1])
    rod = random.choice([0]) # rate of disappearance
    rod = int(rod*N)
    print(trials, N, contrast, speed)
    ##################
    ## xys position for neatly arranged dots
    # xys = []
    # for m in range(2, int(np.log2(N)+1)):
    #     pos = [[(m - 1) / np.log2(N) * (np.cos(i)), (m - 1) / np.log2(N) * (np.sin(i))] for i in np.linspace(0, 2 * np.pi, 2 ** m)]
    #     xys = xys + pos
    # xys = np.array(xys[:N])
    # print(xys.shape, N)
    #################
    dot_stim1 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=n1, sizes=stim_size,
                                                 xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                 opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                 elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
    dot_stim2 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=n2, sizes=stim_size,
                                                 xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                 opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                 elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)

    theta = direction*np.radians(angle)  # could use radians if you like
    rot_mat1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot_mat2 = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_track.fly_court_pos(contours)

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
    t1=0
    while (True):
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        # image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=100, max_size = 4000)

        fly_ori = ellipse[2]
        fly_turn = fly_ori - fly_ori_last

        if fly_turn > 90:
            fly_turn = -(180 - fly_turn)
        elif fly_turn < -90:
            fly_turn = 180 + fly_turn

        fly_frame_ori = fly_frame_ori_last + fly_turn

        if len(pos) != 1:
            frames_lost += 1  # counts number of frames in which the location of fly could not be determined
            print('fly lost')
            pos.append(position)
            if frames_lost > 1000:  # if the number of lost frames is more than 15, abort experiment
                print('Too many frames are being lost')
                misc.send_telegram_message('Fly lost, aborting Experiment!')
                break
        else:
            frames_lost = 0

        x = (pos[0][0] - 512) / w_ratio + w_rem
        y = (pos[0][1] - 512) / h_ratio + h_rem
        dot_stim1.fieldPos = [[x, y]]
        dot_stim2.fieldPos = [[x, y]]

        if frame_number == 0:
            # camera.TimestampLatch()
            # t1 = camera.TimestampLatchValue.GetValue()
            dir = 0
        elif frame_number < stimulus_duration * framerate:
            dir = 0
        elif frame_number >= stimulus_duration * framerate and frame_number < 2*stimulus_duration * framerate:
            dir = direction
            if rod != 0:
                index1 = random.sample(range(n1), rod)
                if n2==0:
                    pass
                else:
                    index2 = random.sample(range(n2), int(rod*(n2/n1)))
                    dot_stim2.xys[index2] = np.stack((np.subtract(2*np.random.random_sample(int(rod*(n2/n1))), 1),
                                                      np.subtract(2*np.random.random_sample(int(rod*(n2/n1))), 1)), axis=1)
                dot_stim1.xys[index1] = np.stack((np.subtract(2*np.random.random_sample(rod), 1),
                                                  np.subtract(2*np.random.random_sample(rod), 1)), axis=1)
            dot_stim1.xys = np.dot(dot_stim1.xys, rot_mat1)
            dot_stim2.xys = np.dot(dot_stim2.xys, rot_mat2)
        elif frame_number >= 2*stimulus_duration * framerate and frame_number < 3*stimulus_duration:
            dir = 0
        elif frame_number >= 3*stimulus_duration and frame_number < 4*stimulus_duration:
            dir = (-1) * direction
            if rod != 0:
                index1 = random.sample(range(n1), rod)
                if n2 == 0:
                    pass
                else:
                    index2 = random.sample(range(n2), int(rod*(n2/n1)))
                    dot_stim2.xys[index2] = np.stack((np.subtract(2*np.random.random_sample(int(rod*(n2/n1))), 1),
                                                      np.subtract(2*np.random.random_sample(int(rod*(n2/n1))), 1)), axis=1)
                dot_stim1.xys[index1] = np.stack((np.subtract(2*np.random.random_sample(rod), 1),
                                                  np.subtract(2*np.random.random_sample(rod), 1)), axis=1)
            ## exchange the rotation matrices to flip the direction
            dot_stim1.xys = np.dot(dot_stim1.xys, rot_mat2)
            dot_stim2.xys = np.dot(dot_stim2.xys, rot_mat1)
        elif frame_number >= 4*stimulus_duration:
            stimulus_inst = objects.stim_inst([stimulus, coherence, rod], direction, N, speed,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_stim_start,
                                          frame_number, video_frame, contrast, stim_size)
            # stimulus_inst.stim_attributes = [wheel_stim.mask,closedloop_gain]
            exp_number = len(total_data['exp_params'])
            total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
            pickle.dump(stimulus_inst.__dict__, f_pickle)
            break
        dot_stim1.draw()
        dot_stim2.draw()
        win.flip()  # slowest part of the algorithm,takes ~10ms
        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp - t1, dir])
        cropped_image = cv_image[int(pos[0][1]) - 30:int(pos[0][1]) + 30, int(pos[0][0]) - 30:int(pos[0][0]) + 30]
        if cropped_image.shape == (60, 60):
            pass
        else:
            print("cropping failure...")
            cropped_image = filler_frame
        writer_small.writeFrame(cropped_image)
        cv2.imshow('frame', cropped_image)
        cv2.waitKey(1)
        position = pos[0]
        fly_ori_last = fly_ori
        fly_frame_ori_last = fly_frame_ori
        frame_number = frame_number + 1
        video_frame = video_frame + 1
        t1 = timestamp
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
    os.remove(files_path_json.strip('.json') + 'copy' + '.json')

misc.send_telegram_message('Experiment Done')
exit()

