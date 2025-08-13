import multiprocessing
# multiprocessing.set_start_method('spawn')
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from queue import LifoQueue

import cv2
# my modules
import experiment_parameters_dialog_box
from initialize_experiment_files import *
import fly_track
import image_methods
import numpy as np
import time
import glob
global num_flies
num_flies = 2
global framerate
framerate = 60  ## this is the refresh rate of the projector (which is 60Hz), NOT the camera (which is 150FPS)
global duration_of_experiment # in minutes
duration_of_experiment = 60
global interstimulus_interval # in seconds
interstimulus_interval = 27
num_trials = int(duration_of_experiment * 150 / 30) ## here the 150 is the frame rate of the camera and 30 is the duration of each trial in seconds
global w_ratio, w_rem, h_ratio, h_rem
w_ratio = 512
w_rem = 0
h_ratio = 512
h_rem = 0

class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)


def present_stimulus(queue1, queue2, files_path_pickle, files_path_json, num_trials, Dir, experiment_data, stimulus_type, total_data):
    import stimulus_presentation
    f_json = open(files_path_json, 'w')  ## open the json file ## important : data is not appended but re-written at the end of the experiment
    stimulus_position_file = open(files_path_json[:-5]+'_stimulus_position.csv', 'w')
    writer_csv_stimulus = csv.writer(stimulus_position_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer_csv_stimulus.writerow(['stimulus_position_x', 'stimulus_position_y'])
    f_pickle = open(files_path_pickle, 'ab')
    pickle.dump(total_data, f_pickle)
    win = stimulus_presentation.initialise_projector_window(Dir)
    randomized_trial_parameters = stimulus_presentation.get_stimulus_combinations(stimulus_type, num_trials, experiment_data)
    randomized_trial_parameters = randomized_trial_parameters.assign(stimulus_name = [stimulus_type] * randomized_trial_parameters.shape[0])
    randomized_trial_parameters.to_csv(os.path.join(Dir, '{}_{}_{}_stimulus_conditions.csv'.format(experiment_data['fly_line'], experiment_data['serial_no'], experiment_data['remark'])))
    print(Dir)
    # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
    print('trying to get data from first image..', flush=True)
    pos, fly_turn, fly_frame_ori_last = queue1.get()
    print(fly_frame_ori_last)

    trials = 0
    video_frame = 0
    time_exp_start = time.process_time()
    while (video_frame < duration_of_experiment * 60 *framerate):
        print(trials)
        # initialize stimulus
        stimulus = stimulus_presentation.initiliaze_stimulus(stimulus_type, win, randomized_trial_parameters.iloc[trials], fly_frame_ori_last)
        frame_number = 0
        time_stim_start = time.process_time()
        while(True):
            pos, fly_turn, fly_frame_ori_last = queue1.get()
            x = pos[0][0]
            y = pos[0][1]
            # update stimulus
            if frame_number < interstimulus_interval * framerate:
                # updates and draw stimulus and returns values to be sent to imaging process(to be written in csv file)
                stimulus_features = stimulus.update_default([x, y], fly_turn, fly_frame_ori_last)
            elif frame_number >= interstimulus_interval * framerate and frame_number < (interstimulus_interval + experiment_data['stimulus_duration']) * framerate:
                # updates and draw stimulus and returns values to be sent to imaging process(to be written in csv file)
                stimulus_features = stimulus.update_with_motion([x, y], fly_turn, fly_frame_ori_last, randomized_trial_parameters.iloc[trials], frame_number)
            elif frame_number >= (interstimulus_interval + experiment_data['stimulus_duration']) * framerate:
                # stimulus_inst = objects.stim_inst(stimulus.__dict__, time.process_time() - time_exp_start, time.process_time() - time_stim_start, frame_number, video_frame)
                stimulus_inst = stimulus.get_stimulus_features()
                stimulus_inst.update(exp_time=time.process_time() - time_exp_start, stim_time=time.process_time() - time_stim_start,
                                                            off_frames=video_frame, on_frames=video_frame-2 * experiment_data['stimulus_duration'] * framerate)
                exp_number = len(total_data['exp_params'])
                total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst)
                pickle.dump(stimulus_inst, f_pickle)
                break
            win.flip()
            writer_csv_stimulus.writerow([x, y])
            try:
                queue2.put(stimulus_features + [frame_number, video_frame], block=False)
            except queue2.Full:
                print('Queue2 is full')
            frame_number+=1
            video_frame+=1
        trials+=1
    try:
        queue2.put([-1, -1, -1], block=False) # send -1, -1, -1 to signal end of experiment
    except queue2.Full:
        print('Queue2 is full')

    json.dump(total_data, f_json)
    f_json.close()
    f_pickle.close()
    return 1


def grab_image(queue1, queue2, csv_file, video_file, stimulus_type):
    # get heades for the csv_file corresponding to the stimulus type
    csv_headers = ['pos_x_m', 'pos_y_m', 'ori_360_m','pos_x_f', 'pos_y_f', 'ori_360_f', 'timestamp', 'frame_number', 'video_frame'] + get_csv_file_headers(stimulus_type)
    # start writers for csv and video file (outputs of experiment)
    csv_file = open(csv_file, 'w')
    writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer_csv.writerow(csv_headers)
    print(video_file)
    writer_small_male = sk.FFmpegWriter(video_file[:-4]+'_male.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset': 'slow'})
    writer_small_female = sk.FFmpegWriter(video_file[:-4]+'_female.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset': 'slow'})
    ## for saving the fulll video (lower quality)
    writer_full = sk.FFmpegWriter(video_file[:-4]+'_full.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset': 'fast'})
    
    ##Intitilize camera and select ROI
    camera = image_methods.initialise_camera("ptgrey")

    ##user finds the arena, by selecting ROI
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    arena = cv2.selectROI(image, False)
    loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
    size = arena[2] // 2

    small_fly, large_fly = fly_track.find_size(camera, loc, size)

    print(max(small_fly), min(small_fly))
    print(max(large_fly), min(large_fly))

    male_min_cutoff = min(small_fly) - 30
    female_min_cutoff = min(large_fly) - 20
    male_max_cutoff = max(small_fly) + 20
    female_max_cutoff = max(large_fly) + 30
    double_min_cutoff = max(female_max_cutoff + 30, male_min_cutoff + female_min_cutoff - 50)
    print(male_min_cutoff, female_min_cutoff, male_max_cutoff, female_max_cutoff, double_min_cutoff)
    
    male_x, male_y, female_x, female_y = fly_track.find_male(camera, 60, loc, size)
    male_position = [male_x, male_y]
    female_position = [female_x, female_y]

    # pos = [[200, 200]]
    # fly_ori = 0
    male_fly_ori = 0
    fly_ori_last = male_fly_ori
    fly_frame_ori_last = male_fly_ori
    position= male_position

    try:
        queue1.put([[male_position], 0, fly_frame_ori_last], block=False)
    except queue1.Full:
        print('Queue1 is full')

    filler_frame = np.zeros((100, 100))
    print('starting to take images', flush=True)

    frame_number = -1
    video_frame = -1
    flies_touching = 0
    fly_lost = 0
    last_position = [male_position, female_position]
    last_ori = [0, 0]
    while (True):  # this unit(of experiment) is repeated #  stimulus is not re-defined inside this loop ## only parameters are changed ##
        image, male_position, female_position, male_fly_ori, female_fly_ori, distance_between_flies, timestamp, area, flies_touching, fly_lost = \
        fly_track.get_male_female_postion_and_orientaion(camera, loc, size, last_position, last_ori, female_max_cutoff, female_min_cutoff, 
                                           male_max_cutoff, male_min_cutoff, double_min_cutoff, flies_touching, fly_lost)

        # pos = [[200, 200]]
        # fly_ori = 0
        fly_ori_last = male_fly_ori
        fly_frame_ori_last = male_fly_ori
        fly_turn = male_fly_ori - fly_ori_last

        if fly_turn > 90:
            fly_turn = -(180 - fly_turn)
        elif fly_turn < -90:
            fly_turn = 180 + fly_turn
        fly_frame_ori = fly_frame_ori_last + fly_turn

        try:
            queue1.put([[male_position], fly_turn, fly_frame_ori], block=False)
        except queue1.Full:
            print('Queue1 is full')
        try:
            info_from_presentation = queue2.get(block=False)
            frame_number = info_from_presentation[-2]
            video_frame = info_from_presentation[-1]
            if frame_number == -1 & video_frame == -1:
                break
        except:
            # print('error')
            info_from_presentation = []
            frame_number, video_frame = frame_number, video_frame

        writer_csv.writerow([male_position[0], male_position[1], male_fly_ori, female_position[0], female_position[1], female_fly_ori,
                             timestamp, frame_number, video_frame] + info_from_presentation[:-2])
        
        cropped_image_male = image[male_position[1] - 50:male_position[1] + 50, male_position[0] - 50:male_position[0] + 50]
        cropped_image_female = image[female_position[1] - 50:female_position[1] + 50, female_position[0] - 50:female_position[0] + 50]
        
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

        ## saving full video
        resized_image = cv2.resize(image, (250, 250)) ## resizing the image to 250x250, to save time (the real bottleneck) and space
        writer_full.writeFrame(resized_image)

        position = male_position
        fly_ori_last = male_fly_ori
        fly_frame_ori_last = fly_frame_ori
        last_position = [male_position, female_position]
        last_ori = [male_fly_ori, female_fly_ori]

    cv2.destroyAllWindows()
    writer_small_male.close()
    writer_small_female.close()
    writer_full.close()
    csv_file.close()
    return 1

if __name__ == '__main__':
    # get the type of stimulus e.g., full pinwheel, half pinwheel, random dots etc.
    stimulus_type = experiment_parameters_dialog_box.get_stimulus_type()
    # # get fly features, this is only for courtship experiments
    # fly_features = experiment_parameters_dialog_box.courtship_info()
    # collect experiment parameters like fly strain, age, stimulus conditions etc. depends on the type of stimulus
    experiment_data = experiment_parameters_dialog_box.get_experiment_and_stimulus_parameters(stimulus_type)
    # set the names of all output files
    total_data, Dir, csv_file, video_file, files_path_pickle, files_path_json = \
        initialize_files(experiment_data['fly_line'], experiment_data['fly_sex'], experiment_data['fly_age'], experiment_data['serial_no'], experiment_data['remark'])

    manager = MyManager()
    manager.start()
    queue1 = manager.LifoQueue() # for sending fly position, fly turn and fly_ori from imaging to stimulus
    queue2 = manager.LifoQueue() # for sending direction and frame number from stimulus to imaging

    stimulus = Process(target=present_stimulus, args=(queue1, queue2, files_path_pickle, files_path_json, num_trials, Dir, experiment_data, stimulus_type, total_data))
    imaging = Process(target=grab_image, args=(queue1, queue2, csv_file, video_file, stimulus_type))

    # start the two processes
    stimulus.start()
    imaging.start()

    # wait for all processes to finish
    stimulus.join()
    imaging.join()
    exit()