import copy
import multiprocessing
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
import mask

from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.managers import BaseManager
from queue import LifoQueue


class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)

'''
This script produces a pinwheel that is centered on the head of the animal and
the pinwheel rotates with the turns made by the fly
the pinwheel has two independent sections that rotate such that artificial translational optic flow is generated
'''


def translation_to_angle(distance):
    stim_size=1.5
    radius = stim_size*256
    if distance < 2:
        return 5*(distance/radius)*(180/math.pi)
    else:
        return 0

##function for generating mask
def ROI(image, x, y):
    height, width = image.shape
    loc1 = x
    size1 = y
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, tuple(loc1), size1, 1, thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data


def present_stimulus(queue1, queue2, fly, files_path_pickle, files_path_json, trial_num, filename, Dir, stimulus_duration):
    print('Entrered stimulus presentation...')
    f_pickle = open(files_path_pickle, 'ab')
    date_time = datetime.datetime.now()
    # video parameters
    framerate = 150
    resolution = (720, 720)
    vid = objects.video(resolution, framerate)  ## vid object defined here
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
            print('There is something wrong with the data, please check manually', flush=True)
            exit()
        elif code == -2:
            print('Incorrect folder selected, fly data has changed', flush=True)
            exit()
        elif code == 1 or code == 2:
            total_data = {**fly.__dict__, **{'exp_params': existing_data['exp_params']}}  # fly_data added to the dictionary
        f_json_read.close()
    else:
        total_data = {**fly.__dict__, **{'exp_params': []}}  # fly_data added to the dictionary

    f_json = open(files_path_json, 'w')  ## open the json file ## important : data is not appended but re-written at the end of the experiment
    pickle.dump(fly.__dict__, f_pickle)  ## start writing to the pickle file, it acts as a temporary storage, in case of ant failure in writing to the json file

    ## experiment object defined here
    experiment = objects.exp([date_time.year, date_time.month, date_time.day],
                             [date_time.hour, date_time.minute, date_time.second, date_time.microsecond],1,1,filename)
    total_data['exp_params'].append(
        {**vid.__dict__, **experiment.__dict__, **{'stim_params': []}})  ## experiment data added to the dictionary
    pickle.dump({**vid.__dict__, **experiment.__dict__}, f_pickle)

    win = misc.initialise_projector_window(Dir)
    time_exp_start = time.clock()
    trials=0
    w_ratio = 500
    w_rem = 0
    h_ratio = 500
    h_rem = 0
    print('starting stimulus presentation', flush=True)
    video_frame = 0
    while (trials < trial_num):
        a = [1, -1]
        direction = random.choice(a)
        stim_size = 1.5  # random.choice(size_stim)
        closedloop_gain_rotation = random.choice([1])
        closedloop_gain_translation = random.choice([0])

        offset_translation = 3
        bars1 = random.choice([6])
        Hz1 = random.choice([10])
        contrast1 = random.choice([25])
        angle1 = (360 * Hz1) / (framerate * bars1)

        bars4 = bars3 = bars2 = bars1
        Hz4 = Hz3 = Hz2 = Hz1
        contrast4 = contrast3 = contrast2 = contrast1
        angle4 = angle3 = angle2 = angle1

        # bars2 = random.choice([6])
        # Hz2 = random.choice([2])
        # contrast2 = random.choice([100])
        # angle2 = (360 * Hz2) / (framerate * bars2)
        # angle2 = (360 * Hz2) / (framerate * bars2)

        stimulus = 'sine_pinwheel'
        stimulus_image1 = stimulus + '_' + str(contrast1) + '_' + str(bars1) + '.png'
        stimulus_image1 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image1)

        stimulus_image2 = stimulus + '_' + str(contrast2) + '_' + str(bars2) + '.png'
        stimulus_image2 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image2)

        stimulus_image3 = stimulus + '_' + str(contrast3) + '_' + str(bars3) + '.png'
        stimulus_image3 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image3)

        stimulus_image4 = stimulus + '_' + str(contrast4) + '_' + str(bars4) + '.png'
        stimulus_image4 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image4)

        mask_library_ori = np.load('mask_data_180.npy')

        wheel_stim1 = psychopy.visual.ImageStim(
            win=win,
            image=stimulus_image1,
            mask='circle',
            pos=(0, 0),
            ori=0,  # 0 is vertical,positive values are rotated clockwise
            size=stim_size,
        )
        wheel_stim1.autoDraw = False

        wheel_stim2 = psychopy.visual.ImageStim(
            win=win,
            image=stimulus_image2,
            mask='circle',
            pos=(0, 0),
            ori=0,  # 0 is vertical,positive values are rotated clockwise
            size=stim_size,
        )
        wheel_stim2.autoDraw = False

        wheel_stim3 = psychopy.visual.ImageStim(
            win=win,
            image=stimulus_image3,
            mask='circle',
            pos=(0, 0),
            ori=0,  # 0 is vertical,positive values are rotated clockwise
            size=stim_size,
        )
        wheel_stim3.autoDraw = False

        wheel_stim4 = psychopy.visual.ImageStim(
            win=win,
            image=stimulus_image4,
            mask='circle',
            pos=(0, 0),
            ori=0,  # 0 is vertical,positive values are rotated clockwise
            size=stim_size,
        )
        wheel_stim4.autoDraw = False

        # main loop
        # to capture and save images
        # to find the location of fly
        # to project stimulus to the required point
        frame_number = 0
        cv2.destroyAllWindows()
        position = (0, 0)
        filler_frame = np.zeros((60, 60))
        time_stim_start = time.clock()
        ##
        mask_library_list = []
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=90+45+45+45)) #right is the rightmost segment
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=90+45)) #left
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=90+45+45)) #center
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=90)) #rear is the leftmost segment
        ##
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135+90+90)) #right
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135)) #left
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135+90)) #center
        # mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=45))  #rear
        ##
        mask_library_list.append(np.array(mask_library_ori))
        mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=180))
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        print('trying to get data from first image..', flush=True)
        pos, fly_turn, fly_frame_ori_last = queue1.get()

        mask_turn_list = []
        mask_turn_list.append(fly_frame_ori_last)
        mask_turn_list.append(fly_frame_ori_last)
        mask_turn_list.append(fly_frame_ori_last)
        mask_turn_list.append(fly_frame_ori_last)
        ##
        min_angle = 360 // len(mask_library_list[0])
        # which side should the stimulus appear on
        # stimulus_side = random.choice(['right_left', 'center_right', 'left_center', 'center'])
        stimulus_side = random.choice(['right', 'right_left', 'left', 'right_left'])
        # stimulus_side = random.choice(['right', 'left'])
        # stimulus_side = random.choice(['center_rear', 'center', 'rear'])
        # stimulus_side = random.choice(['left_center', 'left', 'center'])
        # stimulus_side = random.choice(['right_left', 'center_right', 'left_center'])
        trans_direction = random.choice([0])
        point_of_expansion = random.choice([0])
        print(trials, direction, stimulus_side, flush=True)
        syn_or_anti = random.choice([1, -1])
        no_of_sections = 2
        frames_lost = 0
        fly_frame_ori = fly_frame_ori_last

        frame_number = 0
        while (True):  # this unit(of experiment) is repeated #  stimulus is not re-defined inside this loop ## only parameters are changed ##
            pos, fly_turn, fly_frame_ori_last = queue1.get()
            x = (pos[0][0] - 512) / w_ratio + w_rem
            y = (pos[0][1] - 512) / h_ratio + h_rem
            wheel_stim1.pos = [[x, y]]
            wheel_stim2.pos = [[x, y]]
            wheel_stim3.pos = [[x, y]]
            wheel_stim4.pos = [[x, y]]

            distance = math.sqrt((pos[0][0] - position[0]) ** 2 + (pos[0][1] - position[1]) ** 2)
            translation_angle = translation_to_angle(distance)

            if frame_number == 0:
                dir = 0
            elif frame_number < stimulus_duration * framerate:
                dir = 0
            elif frame_number >= stimulus_duration * framerate and frame_number < 2 * stimulus_duration * framerate:
                dir = direction
            elif frame_number >= 2 * stimulus_duration * framerate and frame_number < 3 * stimulus_duration * framerate:
                dir = 0
            elif frame_number >= 3 * stimulus_duration * framerate and frame_number < 4 * stimulus_duration * framerate:
                dir = (-1) * direction
            elif frame_number >= 4 * stimulus_duration * framerate:
                stimulus_inst1 = objects.stim_inst([stimulus, stimulus_image1, list(win.color), stimulus_side, point_of_expansion], direction, bars1, Hz1,
                                                   time.clock() - time_exp_start,
                                                   time.clock() - time_stim_start,
                                                   frame_number, video_frame, contrast1, stim_size)
                stimulus_inst2 = objects.stim_inst([stimulus, stimulus_image2, list(win.color), stimulus_side, point_of_expansion], direction, bars2, Hz2,
                                                   time.clock() - time_exp_start,
                                                   time.clock() - time_stim_start,
                                                   frame_number, video_frame, contrast2, stim_size)
                stimulus_inst1.stim_attributes = ['semi-circle', closedloop_gain_rotation, closedloop_gain_translation, trans_direction]
                stimulus_inst2.stim_attributes = ['semi-circle', closedloop_gain_rotation, closedloop_gain_translation, trans_direction]
                exp_number = len(total_data['exp_params'])
                total_data['exp_params'][exp_number - 1]['stim_params'].append([stimulus_inst1.__dict__, stimulus_inst2.__dict__])
                pickle.dump(stimulus_inst1.__dict__, f_pickle)
                pickle.dump(stimulus_inst2.__dict__, f_pickle)
                break

            '''
            total_angle = a + bx + c + dy where a = rotn_offset, c = trans_offset, b = rotn_gain, d = trans_gain
                        x = fly_turn and y = distance
            closedloop components are not dependent on the value of dir
            non-closedloop or offset components need to multiplied with dir
            the change in mask should only consider translational motion
            '''
            dir1 = 0
            dir2 = 0
            dir3 = 0
            dir4 = 0
            if stimulus_side == 'left':
                dir2 = dir * 1
            elif stimulus_side == 'right':
                dir1 = dir * 1
            elif stimulus_side == 'center':
                dir3 = dir * 1
            elif stimulus_side == 'rear':
                dir4 = dir * 1
            elif stimulus_side == 'right_left':
                dir2 = dir * 1
                dir1 = dir * syn_or_anti
            elif stimulus_side == 'center_right':
                dir3 = dir * 1
                dir1 = dir * syn_or_anti
            elif stimulus_side == 'left_center':
                dir2 = dir * 1
                dir3 = dir * syn_or_anti
            elif stimulus_side == 'center_rear':
                dir3 = dir * 1
                dir4 = dir * syn_or_anti
            elif stimulus_side == 'right_rear':
                dir1 = dir * 1
                dir4 = dir * syn_or_anti
            elif stimulus_side == 'left_rear':
                dir2 = dir * 1
                dir4 = dir * syn_or_anti

            dir_list = [dir1, dir2, dir3, dir4]
            angle_list = [angle1, angle2, angle3, angle4]
            wheel_stim_list = [wheel_stim1, wheel_stim2, wheel_stim3, wheel_stim4]
            # dir_list = [dir1, dir2]
            # angle_list = [angle1, angle2]
            # wheel_stim_list = [wheel_stim1, wheel_stim2]
            for k in range(no_of_sections):
                wheel_stim_list[k].ori = (wheel_stim_list[k].ori + ((dir_list[k] * angle_list[k]) - (closedloop_gain_rotation * fly_turn)) + (
                            (closedloop_gain_translation * translation_angle)
                            + (trans_direction * offset_translation))) % 360
                # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
                mask_turn_list[k] = (mask_turn_list[k] + (closedloop_gain_translation * translation_angle + trans_direction * offset_translation) + (dir_list[k] * angle_list[k])) % 360
                wheel_stim_list[k].mask = mask_library_list[k][mask.angle_to_index((mask_turn_list[k] + point_of_expansion) % 360, min_angle)]

            for k in range(no_of_sections):
                wheel_stim_list[k].draw()
            win.flip()  # slowest part of the algorithm,takes ~10ms

            try:
                queue2.put([dir1, dir2, dir3, dir4, frame_number, video_frame], block=False)
            except queue2.Full:
                print('Queue2 is full')

            position = pos[0]
            fly_frame_ori_last = fly_frame_ori
            frame_number = frame_number + 1
            video_frame = video_frame + 1
        trials+=1
    queue2.put(None)
    json.dump(total_data, f_json)
    f_json.close()
    f_pickle.close()

    if os.path.isfile(
            files_path_json.strip('.json') + 'copy' + '.json'):  ## delete the json file created at the beginning as a copy
        os.remove(files_path_json.strip('.json') + 'copy' + '.json')

    misc.send_telegram_message('Experiment Done')
    return 1


def grab_image(queue1, queue2, filename):
    print('Started imaging..')
    csv_file = filename + '.csv'
    if os.path.isfile(csv_file):
        print('This file already exists, please add a remark or identifier to uniquely identify this video', flush=True)
        reply = message_box.data_change_error('Warning : File already exists', 'Proceed?')
        if reply == 1:
            filename = misc.change_file_name(filename)
            csv_file = filename + '.csv'
        else:
            exit()
    csv_file = open(csv_file, 'w')
    writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer_csv.writerow(['pos_x', 'pos_y', 'ori','timestamp', 'direction_left', 'direction_right', 'direction_center', 'direction_rear', 'trial_frame', 'video_frame'])
    writer_small = sk.FFmpegWriter(filename + '.avi', inputdict={'-r': '150'}, outputdict={'-vcodec': 'libx264',
                                                                                          '-crf': '0',
                                                                                          '-preset': 'slow'})
    # camera intialisation
    camera = image_methods.initialise_camera("ptgrey")
    print('Camera intitialised..')

    # first image taken for pre-processing
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # user finds the arena, by selecting ROI
    print('Find ROI..')
    arena = cv2.selectROI(image, False)
    loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
    size = arena[2] // 2

    """find the first value for the fly angle"""
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    # image, timestamp = image_methods.grab_image(camera, 'basler')
    cv_image = ROI(image, loc, size)
    ret, diff_img = cv2.threshold(cv_image, 70, 255, cv2.THRESH_BINARY)
    this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pos, ellipse, cnt = fly_track.fly_court_pos(contours)

    fly_ori = 0
    fly_ori_last = ellipse[2]
    fly_frame_ori_last = ellipse[2]
    print('grabbed the first image', flush=True)
    try:
        queue1.put([pos, 0, fly_frame_ori_last], block=False)
    except queue1.Full:
        print('Queue1 is full')

    filler_frame = np.zeros((60, 60))
    print('starting to take images', flush=True)
    frames_lost = 0
    while (True):  # this unit(of experiment) is repeated #  stimulus is not re-defined inside this loop ## only parameters are changed ##
        image, timestamp = image_methods.grab_image(camera, 'ptgrey')
        # image, timestamp = image_methods.grab_image(camera, 'basler')
        cv_image = ROI(image, loc, size)
        ret, diff_img = cv2.threshold(cv_image, 60, 255, cv2.THRESH_BINARY)
        this, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pos, ellipse, cnt = fly_track.fly_court_pos(contours, size_cutoff=200, max_size=4000)

        fly_ori = ellipse[2]
        fly_turn = fly_ori - fly_ori_last

        if fly_turn > 90:
            fly_turn = -(180 - fly_turn)
        elif fly_turn < -90:
            fly_turn = 180 + fly_turn

        fly_frame_ori = fly_frame_ori_last + fly_turn

        if len(pos) != 1:
            frames_lost += 1  # counts number of frames in which the location of fly could not be determined
            pos.append(position)
            if frames_lost > 300:  # if the number of lost frames is more than 15, abort experiment
                print('Too many frames are being lost', flush=True)
                exit()
        else:
            frames_lost = 0
        try:
            queue1.put([pos, fly_turn, fly_frame_ori], block=False)
        except queue1.Full:
            print('Queue1 is full')
        try:
            info_from_presentation = queue2.get(block=False)
            if info_from_presentation == None:
                break
            else:
                dir1, dir2, dir3, dir4, frame_number, video_frame = info_from_presentation
        except:
            dir1, dir2, dir3, dir4, frame_number, video_frame = 0,0,0,0,-1, -1

        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp, dir2, dir1, dir3, dir4, frame_number, video_frame])
        cropped_image = cv_image[int(pos[0][1]) - 30:int(pos[0][1]) + 30, int(pos[0][0]) - 30:int(pos[0][0]) + 30]
        if cropped_image.shape == (60, 60):
            pass
        else:
            print("cropping failure...", flush=True)
            cropped_image = filler_frame
        writer_small.writeFrame(cropped_image)
        cv2.imshow('frame2', cropped_image)
        cv2.waitKey(1)
        position = pos[0]
        fly_ori_last = fly_ori
        fly_frame_ori_last = fly_frame_ori

    cv2.destroyAllWindows()
    writer_small.close()
    csv_file.close()

    return 1

if __name__ == '__main__':
    # exp_info = info()
    # # fly_parameters
    # strain = exp_info[0]  # fly strain
    # serial = exp_info[3]  # fly_id
    # sex = exp_info[1]
    # age = int(exp_info[2])
    # fly = objects.fly(strain, serial, sex, age)  ## fly object defined here
    # if exp_info[6] == '_':
    #     remark = ''
    # else:
    #     remark = '_' + exp_info[6]
    # stimulus_duration = int(exp_info[7])
    # bg = int(exp_info[4])
    # fg = int(exp_info[5])

    strain = 'wt'  # fly strain
    serial = '1'  # fly_id
    sex = 'F'
    age = 1
    fly = objects.fly(strain, serial, sex, age)  ## fly object defined here
    remark = 'o'
    stimulus_duration = 5
    bg=1
    fg=1

    # experiment parameters
    num_bars = [4, 6, 12]
    freq = [1, 2, 5, 10, 15, 20]
    size_stim = [0.25, 0.5, 0.75, 1.0]
    bright_bars = np.linspace(0.0, 1.0, 11)
    dark_bars = np.linspace(0.0, 1.0, 11)
    date_time = datetime.datetime.now()
    Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
    if os.path.isdir(Dir) == False:  ## checks if this folder exists
        print('Folder name incorrect', flush=True)
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
    filename = strain + '_' + serial + '_' + date_string + remark
    filename = r'{}\{}'.format(Dir_new, filename)

    trial_num=10
    manager = MyManager()
    manager.start()
    queue1 = manager.LifoQueue() # for sending fly position, fly turn and fly_ori from imaging to stimulus
    queue2 = manager.LifoQueue() # for sending direction and frame number from stimulus to imaging
    stimulus = Process(target=present_stimulus, args=(queue1, queue2,fly, files_path_pickle, files_path_json, trial_num, filename, Dir, stimulus_duration))
    imaging = Process(target=grab_image, args=(queue1, queue2,filename))
    stimulus.start()
    imaging.start()
    # wait for all processes to finish
    stimulus.join()
    imaging.join()
    exit()