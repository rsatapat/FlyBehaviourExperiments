import os
import datetime
import csv
import shutil
import copy
import json
import misc
import pickle
import objects
import message_box
import skvideo.io as sk
from tkinter import filedialog, Tk

global csv_file
global f_json
global f_pickle
import shelve

def file_chooser():
    default_filename = 'most_recent_folder.dat'
    if os.path.exists(default_filename):
        db = shelve.open('most_recent_folder')
        Dir = db['folder']
    else:
        db = shelve.open('most_recent_folder')
        Dir = os.getcwd()
    application_window = Tk()
    # Ask the user to select a folder.
    answer = filedialog.askdirectory(parent=application_window, initialdir=Dir, title="Please select a folder:")
    db.update(dict(folder=answer))
    db.close()
    return answer

def initialize_files(strain, sex, age, serial, remark):
    fly = objects.fly(strain, serial, sex, age)
    Dir = file_chooser()  ## choose the destination folder ## the folder which contains all the flies of this strain
    if os.path.isdir(Dir) == False:  ## checks if this folder exists
        print('Folder name incorrect')
        exit()
    date_time = datetime.datetime.now()
    date_string = ''
    for i in [date_time.year, date_time.month, date_time.day]:
        date_string = date_string + str(i)

    Dir_name = fly.strain + '_' + fly.id
    Dir_new = os.path.join(Dir, Dir_name)

    if os.path.isdir(Dir_new) == False:  ## checking to see of the fly already has a folder or not
        os.mkdir(Dir_new)  ## if no such folder exists, make one

    file_name_json = Dir_name + '.json'
    file_name_pickle = Dir_name + '.p'

    files_path_json = os.path.join(Dir_new, file_name_json)
    files_path_pickle = os.path.join(Dir_new, file_name_pickle)
    # f_pickle = open(files_path_pickle, 'ab')

    ## checking to see if a json file with the same name exists
    exists = os.path.isfile(files_path_json)
    if exists:
        shutil.copyfile(files_path_json, files_path_json.strip('.json') + 'copy' + '.json')
        f_json_read = open(files_path_json, 'r')
        existing_data = json.load(f_json_read)
        existing_data_copy = copy.deepcopy(existing_data)  ## create a deep copy of the existing_data
        del existing_data_copy['exp_params']
        code = misc.compare_dicts(fly.__dict__, existing_data_copy)  ## change code here to include only the fly data and not any other data
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


    # pickle.dump(fly.__dict__, f_pickle)  ## start writing to the pickle file, it acts as a temporary storage, in case of any failure in writing to the json file

    filename = fly.strain + '_' + fly.id + '_' + date_string + remark
    filename = os.path.join(Dir_new, filename)
    csv_file = filename + '.csv'
    if os.path.isfile(csv_file):
        print('This file already exists, please add a remark or identifier to uniquely identify this video')
        reply = message_box.data_change_error('Warning : File already exists', 'Proceed?')
        if reply == 1:
            filename = misc.change_file_name(filename)
            csv_file = filename + '.csv'
        else:
            exit()
    video_file = filename + '.avi'

    ## experiment object defined here
    framerate = 60
    resolution = (720, 720)
    vid = objects.video(resolution, framerate)
    experiment = objects.exp([date_time.year, date_time.month, date_time.day],
                             [date_time.hour, date_time.minute, date_time.second, date_time.microsecond], filename)
    total_data['exp_params'].append({**vid.__dict__, **experiment.__dict__, **{'stim_params': []}})  ## experiment data added to the dictionary
    # pickle.dump({**vid.__dict__, **experiment.__dict__}, f_pickle)

    return total_data, Dir_new, csv_file, video_file, files_path_pickle, files_path_json

def get_csv_file_headers(stimulus_type):
    if stimulus_type=='Full pinwheel' or stimulus_type=='Full Random Dots' or stimulus_type == 'Flicker pinwheel':
        return ['direction']
    elif stimulus_type=='Half pinwheel' or stimulus_type == 'Half pinwheel ring' or stimulus_type == 'Half pinwheel flicker'\
        or stimulus_type == 'Half pinwheel binocular overlap' or stimulus_type == 'Half random dots' or stimulus_type == 'Half pinwheel oscillate':
        return ['direction_right', 'direction_left'] # changed on 11/11/2023, was left,right before that
    elif stimulus_type=='Quarter pinwheel front':
        return ['direction_center', 'direction_rear'] # the sequence of directios is right, left, center, rear
    elif stimulus_type=='dark and light':
        return ['color'] # the sequence of directios is right, left, center, rear
    
    
def close_files():
    csv_file.close()
    f_json.close()
    f_pickle.close()