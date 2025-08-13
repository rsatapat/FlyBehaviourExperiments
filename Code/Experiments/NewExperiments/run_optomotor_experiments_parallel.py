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
import time

global num_flies
num_flies = 1
global framerate
framerate = 60
global num_trials
num_trials = 120
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
    f_pickle = open(files_path_pickle, 'ab')
    pickle.dump(total_data, f_pickle)
    win = stimulus_presentation.initialise_projector_window(Dir)
    randomized_trial_parameters = stimulus_presentation.get_stimulus_combinations(stimulus_type, num_trials, experiment_data)
    randomized_trial_parameters.to_csv(os.path.join(Dir, '{}_{}_{}_stimulus_conditions.csv'.format(experiment_data['fly_line'], experiment_data['serial_no'], experiment_data['remark'])))
    # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
    print('trying to get data from first image..', flush=True)
    pos, fly_turn, fly_frame_ori_last = queue1.get()
    print(fly_frame_ori_last)

    trials = 0
    video_frame = 0
    time_exp_start = time.process_time()
    while (trials < num_trials):
        print(trials)
        # initialize stimulus
        stimulus = stimulus_presentation.initiliaze_stimulus(stimulus_type, win, randomized_trial_parameters.iloc[trials], fly_frame_ori_last)
        frame_number = 0
        time_stim_start = time.process_time()
        while(True):
            pos, fly_turn, fly_frame_ori_last = queue1.get()
            if fly_turn == -1 and fly_frame_ori_last == -1:
                exit()
            x = pos[0][0]
            y = pos[0][1]
            # update stimulus
            if frame_number < experiment_data['stimulus_duration'] * framerate:
                # updates and draw stimulus and returns values to be sent to imaging process(to be written in csv file)
                stimulus_features = stimulus.update_default([x, y], fly_turn)
            elif frame_number >= experiment_data['stimulus_duration'] * framerate and frame_number < 2 * experiment_data['stimulus_duration'] * framerate:
                # updates and draw stimulus and returns values to be sent to imaging process(to be written in csv file)
                stimulus_features = stimulus.update_with_motion([x, y], fly_turn, randomized_trial_parameters.iloc[trials], frame_number)
            elif frame_number >= 2 * experiment_data['stimulus_duration'] * framerate:
                # stimulus_inst = objects.stim_inst(stimulus.__dict__, time.process_time() - time_exp_start, time.process_time() - time_stim_start, frame_number, video_frame)
                stimulus_inst = stimulus.get_stimulus_features()
                stimulus_inst.update(exp_time=time.process_time() - time_exp_start, stim_time=time.process_time() - time_stim_start,
                                                            off_frames=video_frame, on_frames=video_frame-2 * experiment_data['stimulus_duration'] * framerate)
                exp_number = len(total_data['exp_params'])
                total_data['exp_params'][exp_number - 1]['stim_params'].append(stimulus_inst)
                pickle.dump(stimulus_inst, f_pickle)
                break
            win.flip()

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
    csv_headers = ['pos_x', 'pos_y', 'ori', 'timestamp', 'frame_number', 'video_frame'] + get_csv_file_headers(stimulus_type)
    # start writers for csv and video file (outputs of experiment)
    csv_file = open(csv_file, 'w')
    writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer_csv.writerow(csv_headers)
    print(video_file)
    writer_small = sk.FFmpegWriter(video_file, inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset': 'slow'})

    ##Intitilize camera and select ROI
    camera = image_methods.initialise_camera("ptgrey")

    ##user finds the arena, by selecting ROI
    image, timestamp = image_methods.grab_image(camera, 'ptgrey')
    arena = cv2.selectROI(image, False)
    loc = [arena[0] + arena[2] // 2, arena[1] + arena[3] // 2]
    size = arena[2] // 2

    _, pos, fly_ori, timestamp = fly_track.get_fly_postion_and_orientaion(camera, loc, size)
    # pos = [[200, 200]]
    # fly_ori = 0
    fly_ori_last = fly_ori
    fly_frame_ori_last = fly_ori
    position= pos[0]
    print('grabbed the first image', flush=True)
    try:
        queue1.put([pos, 0, fly_frame_ori_last], block=False)
    except queue1.Full:
        print('Queue1 is full')

    filler_frame = np.zeros((60, 60))
    print('starting to take images', flush=True)
    frames_lost = 0
    frame_number = -1
    video_frame = -1
    missed_stimulus = 0
    while (True):  # this unit(of experiment) is repeated #  stimulus is not re-defined inside this loop ## only parameters are changed ##
        cv_image, pos, fly_ori, timestamp = fly_track.get_fly_postion_and_orientaion(camera, loc, size)
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
                queue1.put([-1, -1, -1], block=False)
                exit()
        else:
            frames_lost = 0
        try:
            queue1.put([pos, fly_turn, fly_frame_ori], block=False)
        except queue1.Full:
            print('Queue1 is full')
        try:
            info_from_presentation = queue2.get(block=False)
            frame_number = info_from_presentation[-2]
            video_frame = info_from_presentation[-1]
            if frame_number == -1 & video_frame == -1:
                break
            missed_stimulus=0 # reset missed stimulus counter
        except:
            missed_stimulus+=1 # increment missed stimulus counter
            if missed_stimulus > 20: # if the stimulus is missed for more than 20 frames, abort video recording
                break
            info_from_presentation = []
            frame_number, video_frame = frame_number, video_frame

        writer_csv.writerow([pos[0][0], pos[0][1], fly_frame_ori, timestamp, frame_number, video_frame] + info_from_presentation[:-2])
        cropped_image = cv_image[int(pos[0][1]) - 30:int(pos[0][1]) + 30, int(pos[0][0]) - 30:int(pos[0][0]) + 30]
        if cropped_image.shape == (60, 60):
            pass
        else:
            print("cropping failure...", flush=True)
            # cropped_image = filler_frame
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
    # get the type of stimulus e.g., full pinwheel, half pinwheel, random dots etc.
    stimulus_type = experiment_parameters_dialog_box.get_stimulus_type()
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