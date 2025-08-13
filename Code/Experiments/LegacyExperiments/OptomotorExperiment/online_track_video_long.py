import copy
import time
import random
import os
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
from Dialog_box import info
from File_chooser import file_chooser
import objects
import fly_track
import image_methods
import misc
import message_box

exp_info = info()

#fly_parameters
strain = exp_info[0] #fly strain
serial = exp_info[3] #fly_id
sex = exp_info[1]
age = exp_info[2]
fly = objects.fly(strain, serial, sex, age)   ## fly object defined here
if exp_info[6] == '_':
    remark = ''
else:
    remark = '_'+exp_info[6]


#video parameters
framerate = 60
resolution = (720,720)
vid = objects.video(resolution,framerate)   ## vid object defined here

#experiment parameters
num_bars = [4, 6, 12]
freq = [1, 2, 5, 10, 15, 20]
bg = int(exp_info[4])
fg = int(exp_info[5])
size_stim = [0.25, 0.5, 0.75, 1.0]
bright_bars = np.linspace(0.0, 1.0, 11)
dark_bars = np.linspace(0.0, 1.0, 11)
date_time = datetime.datetime.now()
stimulus = 'pinwheel'

Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
if os.path.isdir(Dir) == False:  ## checks if this folder exists
    print('Folder name incorrect')
    exit()

date_string = ''
for i in [date_time.year,date_time.month,date_time.day]:
    date_string = date_string + str(i)


Dir_name = strain+str(serial)
Dir_new = r'{}\{}'.format(Dir,Dir_name)

if os.path.isdir(Dir) == False:   ## checking to see of the fly already has a folder or not
    os.mkdir(Dir_new)           ## if no such folder exists, make one

file_name_json = Dir_name + '.json'
file_name_pickle = Dir_name + '.p'

files_path_json = r'{}\{}'.format(Dir_new,file_name_json)
files_path_pickle = r'{}\{}'.format(Dir_new,file_name_pickle)
f_pickle = open(files_path_pickle, 'ab')


## checking to see if a json file with the same name exists
exists = os.path.isfile(files_path_json)
if exists:
    shutil.copyfile(files_path_json, files_path_json.strip('.json') + 'copy' + '.json')
    f_json_read = open(files_path_json, 'r')
    existing_data = json.load(files_path_json)
    existing_data_copy = copy.copy(existing_data)
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

f_json = open(files_path_json, 'w')  ## open the json file ## important : data is not appended but re-written at the end of the experiment
pickle.dump(fly.__dict__,f_pickle)      ## start writing to the pickle file, it acts as a temporary storage, in case of ant failure in writing to the json file



filename = strain + '_' + serial + '_' + date_string + remark
filename = r'{}\{}'.format(Dir_new, filename)
csv_file = filename + '.csv'
if os.path.isfile(csv_file):
    print('This file already exists, please add a remark or identifier to uniquely identify this video')
    reply = message_box.data_change_error('Warning : File already exists', 'Proceed or abort?')
    if reply == 1:
        filename = misc.change_file_name(filename)
        csv_file = filename + '.csv'
csv_file = open(csv_file, 'w+')
writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer_csv.writerow(['pos_x', 'pos_y', 'timestamp'])
writer_small = sk.FFmpegWriter(filename + '.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264',
                                                                                          '-crf': '0',
                                                                                          '-preset': 'slow'})

## experiment object defined here
experiment = objects.exp([date_time.year,date_time.month,date_time.day],[date_time.hour,date_time.minute,date_time.second,date_time.microsecond],bg,fg,filename)


total_data['exp_params'].append({**vid.__dict__, **experiment.__dict__,**{'stim_params':[]}}) ## experiment data added to the dictionary
pickle.dump({**vid.__dict__, **experiment.__dict__},f_pickle)


##function for generating mask
def ROI(image,x,y):
    height,width = image.shape
    loc1 = x
    size1 = y
    circle_img = np.zeros((height,width), np.uint8)
    cv2.circle(circle_img,tuple(loc1),size1,1,thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data


##camera intialisation
camera = image_methods.initialise_camera("ptgrey")


##psychopy stimulus window initialisation
win = psychopy.visual.Window(
    size=[342, 684],
    pos = [1998,-20],
    color = (0,0,0),
    fullscr=False,
    waitBlanking = True
)

win.recordFrameIntervals = True
win.refreshThreshold = 1/60 + 0.004

bars = random.choice(num_bars)
Hz = random.choice(freq)
stim_size = random.choice(size_stim)
contrast = 100
angle = (360 * Hz) / (framerate * bars)
print(bars, angle)

stimulus_image = stimulus + '_' + str(contrast) + '_' + str(bars) + '.png'
stimulus_image = r'{}\{}'.format('D:\Roshan\Project\Stimulus\Generate stimulus', stimulus_image)

wheel_stim = psychopy.visual.ImageStim(
    win=win,
    image=stimulus_image,
    mask='circle',
    pos=(0, 0),
    ori=0,
    size=stim_size,
)
wheel_stim.autoDraw = False


w_ratio = 512
w_rem = 0
h_ratio = 512
h_rem = 0

##first image taken for pre-processing
image, timestamp = camera.grab_image(camera, 'ptgrey')
height,width = image.shape

##user finds the arena, by selecting ROI
arena = cv2.selectROI(image, False)
loc = [arena[0] + arena[2]//2,arena[1] + arena[3]//2]
size = arena[2]//2

cv_image = ROI(image,loc,size)

ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)

this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
pos, ellipse, cnt = fly_track.fly_court_pos(contours)


##main loop
##to capture and save images
##to find the location of fly
##to project stimulus to the required point
frame_number = 0
cv2.destroyAllWindows()
position = (0, 0)
filler_frame = np.zeros((60, 60))

## this is where the experiment starts
time_exp_start = time.clock()
## first stimulus instance
time_stim_start = time.clock()

## status of stimulus 0 or 1 = ON or OFF
stim = random.choice([0,1])
frame_lost = 0
while frame_number < 7200*60*60:
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')

        cv_image = ROI(cv_image,loc,size)
        
        ret, diff_img = cv2.threshold(cv_image,60,255,cv2.THRESH_BINARY_INV)
        this,contours,hierarchy = cv2.findContours(diff_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        pos, ellipse, cnt = fly_track.fly_court_pos(contours)

        if len(pos) != 1:
            frames_lost += 1  # counts number of frames in which the location of fly could not be determined
            pos.append(position)
            if frames_lost > 15:  # if the number of lost frames is more than 15, abort experiment
                print('Too many frames are being lost')
                exit()
        else:
            frames_lost = 0

            
        x = (pos[0][0]-512)/w_ratio + w_rem
        y = (pos[0][1]-512)/h_ratio + h_rem 
        wheel_stim.pos = [[x,y]]
        wheel_stim.pos = [[x,y]]
        writer_csv.writerow([pos[0][0], pos[0][1], timestamp, direction])
        
        if frame_number < stim_time*framerate:
            if stim == 0:
                stim_inst = objects.stim_inst([0, stimulus_image], 0, bars, Hz,
                                              time.clock() - time_exp_start,
                                              time.clock() - time_stim_start,
                                              0, frame_number, contrast, stim_size)
            else:
                wheel_stim.ori = wheel_stim.ori + direction*angle
                wheel_stim.ori = wheel_stim.ori - direction*angle
                wheel_stim.draw()
                wheel_stim.draw()
        else:
            date_time_end = datetime.datetime.now()
            stimulus_inst = objects.stim_inst([stimulus, stimulus_image], direction, bars, Hz,
                                          time.clock() - time_exp_start,
                                          time.clock() - time_stim_start,
                                          0, frame_number, contrast, stim_size)
            stimulus_inst.stim_attributes = [wheel_stim.mask]
            exp_number = len(total_data['exp_params'])
            total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst.__dict__)
            pickle.dump(stimulus_inst.__dict__, f_pickle)
            stim = random.choice([0,1])
            direction = random.choice([0,1, -1])
            stim_time = random.choice(np.linspace(0,60,13))
            frame_number = -1
            time_stim_start = time.clock()
        win.flip()
        
        cropped_image = cv_image[int(pos[0][1])-30:int(pos[0][1])+30,int(pos[0][0])-30:int(pos[0][0])+30]
        if cropped_image.shape == (60,60):
            writer_small.writeFrame(cropped_image)
        else:
            writer_small.writeFrame(filler_frame)
        position = pos[0]
        frame_number = frame_number + 1
       

camera.stopCapture()
camera.disconnect()
cv2.destroyAllWindows()
writer_small.close()
csv_file.close()
json.dump(total_data, f_json)
f_json.close()
f_pickle.close()

exit()



