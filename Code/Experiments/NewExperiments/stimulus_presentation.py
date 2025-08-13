import psychopy.event
import psychopy.visual

import os
import numpy as np
import pandas as pd
import random
import mask

global w_ratio, w_rem, h_ratio, h_rem, destination
w_ratio = 512
w_rem = 0
h_ratio = 512
h_rem = 0

destination = r'C:\Users\rsatapat\PycharmProjects\Project\PythonCodes\Stimulus\Generate stimulus'
if not os.path.exists(destination):
    destination = r'D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus'

class transform_fly_pos():
    def transform_fly_pos(self, x, y):
        x = (x - 512) / w_ratio + w_rem
        y = (y - 512) / h_ratio + h_rem
        return [x, y]


class dark_and_light(transform_fly_pos):
    def __init__(self, win, contrast, framerate, closedloop_gain):
        self.stim_size = 2
        self.name = 'rectangle'
        self.win = win
        self.contrast = contrast
    
    def create_stimulus(self):
        self.stimulus = psychopy.visual.Rect(
            win=self.win,
            width=self.stim_size,
            height=self.stim_size,
            pos=(0, 0),
            color=[self.contrast, self.contrast, self.contrast],
            size=self.stim_size,
        )
        self.stimulus.autoDraw = False

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        # self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        # self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.draw()
        return [0]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        return features


class full_pinwheel_old(transform_fly_pos):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        stimulus = 'sine_pinwheel'
        self.stim_size = 1.0 ## change size to 1.5 for optomotor experiments
        self.stimulus_image = os.path.join(destination, stimulus + '_' + str(int(contrast)) + '_' + str(int(spatial_frequency)) + '.png')
        self.name = 'full_pinwheel'
        self.win = win
        self.contrast = contrast
        self.closedloop_gain = closedloop_gain
        self.temporal_frequency = temporal_frequency
        self.spatial_frequency = spatial_frequency
        self.angle = ((360/self.spatial_frequency) * (self.temporal_frequency))/framerate
    
    def create_stimulus(self):
        self.stimulus = psychopy.visual.ImageStim(
            win=self.win,
            image=self.stimulus_image,
            mask='circle',
            pos=(0, 0),
            ori=0,  # 0 is vertical,positive values are rotated clockwise
            size=self.stim_size,
        )
        self.stimulus.autoDraw = False

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.direction = trial_parameters['direction']
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.ori = (self.stimulus.ori + self.angle*trial_parameters['direction'] - self.closedloop_gain*fly_turn) % 360
        self.stimulus.draw()
        return [trial_parameters['direction']]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('stimulus_image')
        return features


class sinusoidal_pinwheel(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        full_pinwheel_old.__init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        angle = frame_number * 2 * self.speed % 360
        angle = self.amplitude * (np.sin((angle / 180) * np.pi))
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.ori = (self.stimulus.ori + self.angle - self.closedloop_gain*fly_turn) % 360
        self.stimulus.draw()
        return [angle]


class full_flicker(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        full_pinwheel_old.__init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)
        self.name = 'full_flicker'

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        # temporal frequency is the frequency of the flicker, not the speed of rotation
        self.angle = (360/(self.spatial_frequency * 2))
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        if frame_number % (2*self.temporal_frequency) == 0:
            self.stimulus.ori = (self.stimulus.ori + self.angle - self.closedloop_gain*fly_turn) % 360
        elif frame_number % self.temporal_frequency == 0:
            self.stimulus.ori = (self.stimulus.ori - self.angle - self.closedloop_gain*fly_turn) % 360
        else:
            self.stimulus.ori = (self.stimulus.ori - self.closedloop_gain*fly_turn) % 360
        self.stimulus.draw()
        return [self.angle]


class triangular_pinwheel(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        full_pinwheel_old.__init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.angle = self.amplitude - abs(2 * self.amplitude - (self.amplitude * frame_number % (4 * self.amplitude)))
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.ori = (self.stimulus.ori + self.angle - self.closedloop_gain*fly_turn) % 360
        self.stimulus.draw()
        return [self.angle]


class full_pinwheel_new(transform_fly_pos):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        self.stim_size=1
        self.stimulus = psychopy.visual.RadialStim(
            win=win,
            pos=(0, 0),
            size=self.stim_size,
            tex='sin',
            contrast=contrast,
            radialCycles=0,
            angularCycles=spatial_frequency,
            radialPhase=0,
            angularPhase=0,
            interpolate=False,
            autoLog=False,
        )
        self.name = 'full_pinwheel'
        self.win = win
        self.stimulus.autoDraw = False
        self.contrast = contrast
        self.closedloop_gain = closedloop_gain
        self.temporal_frequency = temporal_frequency
        self.spatial_frequency = spatial_frequency
        self.angle = ((360/self.spatial_frequency) * (self.temporal_frequency))/framerate

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.direction = trial_parameters['direction']
        self.stimulus.pos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus.ori = (self.stimulus.ori + self.angle*trial_parameters['direction'] - self.closedloop_gain*fly_turn) % 360
        self.stimulus.draw()
        return [trial_parameters['direction']]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        return features


class segmented_pinwheel_old(transform_fly_pos):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        self.n_segments=n_segments
        self.contrast = list(contrast_list)
        self.closedloop_gain_rotation = list(closedloop_gain_rotation_list)
        self.temporal_frequency = list(temporal_frequency_list)
        self.spatial_frequency = list(spatial_frequency_list)
        self.closedloop_gain_translation = [0, 0]
        self.point_of_expansion = [0, 0]
        self.translation_direction = [0, 0]
        self.name = 'pinwheel'
        self.win = win
        self.stimulus_image = []
        self.angle = []
        for k in range(self.n_segments):
            stimulus = 'sine_pinwheel'
            stimulus_filename = stimulus + '_' + str(int(self.contrast[k])) + '_' + str(int(self.spatial_frequency[k])) + '.png'
            self.stimulus_image.append(os.path.join(destination, stimulus_filename))
            self.angle.append(((360/self.spatial_frequency[k]) * (self.temporal_frequency[k]))/framerate)

    def create_stimulus(self):
        self.stimulus = []
        for k in range(self.n_segments):
            self.stim_size = 1.5
            wheel_stim = psychopy.visual.ImageStim(
                win=self.win,
                image=self.stimulus_image[k],
                mask='circle',
                pos=(0, 0),
                ori=0,  # 0 is vertical,positive values are rotated clockwise
                size=self.stim_size,
            )
            self.stimulus.append(wheel_stim)
            self.stimulus[k].autoDraw = False

    def create_mask(self, fly_frame_ori):
        mask_library_ori = np.load('mask_data_180.npy')
        self.mask_library_list = []
        self.mask_library_list.append(np.array(mask_library_ori))
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=180))
                                 
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        self.mask_turn_list = []
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        min_angle = 360//len(self.mask_library_list[0])
        for k in range(self.n_segments):
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((0 * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*0)
                                                                                                                        + (self.translation_direction[k]*0))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly

            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*0 + self.translation_direction[k]*0) + (0 * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+0)%360, min_angle)]
        for k in range(self.n_segments):
            self.stimulus[k].draw()
        return [0] * self.n_segments

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        '''
        total_angle = a + b.x + c + d.y where a = rotn_offset, c = trans_offset, b = rotn_gain, d = trans_gain
                    x = fly_turn and y = distance
        closedloop components are not dependent on the value of dir
        non-closedloop or offset components need to multiplied with dir
        the change in mask should only consider translational motion
        '''
        translation_angle=0
        min_angle = 360//len(self.mask_library_list[0])
        offset_translation = 3
        self.direction = list(trial_parameters['direction'])
        for k in range(self.n_segments):
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((self.direction[k] * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*translation_angle)
                                                                                                                        + (self.translation_direction[k]*offset_translation))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*translation_angle + self.translation_direction[k]*offset_translation) + (self.direction[k] * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+self.point_of_expansion[0])%360, min_angle)]

        for k in range(self.n_segments):
            self.stimulus[k].draw()

        return trial_parameters['direction']

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('mask_library_list')
        features.pop('mask_turn_list')
        features.pop('stimulus_image')
        return features


class half_pinwheel_old(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        super().__init__(win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori)
        self.name = 'half_pinwheel'


class binocular_overlap(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        super().__init__(win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori)
        self.name = 'binocular_overlap'
        self.overlap_angle = 22
        self.closedloop_gain = 1

    def create_stimulus(self):
        super().create_stimulus()
        self.overlap_wedge1 = psychopy.visual.RadialStim(
                win=self.win,
                pos=(0, 0),
                size=self.stim_size + 0.1,
                tex='sin',
                contrast=0,
                radialCycles=0,
                angularCycles=1,
                color=(1,1,1),
                visibleWedge=[360-self.overlap_angle, 360])
        
        self.overlap_wedge2 = psychopy.visual.RadialStim(
                win=self.win,
                pos=(0, 0),
                size=self.stim_size + 0.1,
                tex='sin',
                contrast=0,
                radialCycles=0,
                angularCycles=1,
                color=(1,1,1),
                visibleWedge=[0, self.overlap_angle])
        
        self.overlap_wedge3 = psychopy.visual.RadialStim(
                win=self.win,
                pos=(0, 0),
                size=self.stim_size + 0.1,
                tex='sin',
                contrast=0,
                radialCycles=0,
                angularCycles=1,
                color=(1,1,1),
                visibleWedge=[360-self.overlap_angle-180, 360-180])
        
        self.overlap_wedge4 = psychopy.visual.RadialStim(
                win=self.win,
                pos=(0, 0),
                size=self.stim_size + 0.1,
                tex='sin',
                contrast=0,
                radialCycles=0,
                angularCycles=1,
                color=(1,1,1),
                visibleWedge=[0+180, self.overlap_angle+180])
    
    def create_mask(self, fly_frame_ori):
        mask_library_ori = np.load('mask_data_180.npy')
        self.mask_library_list = []
        self.mask_library_list.append(np.array(mask_library_ori))
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=180))
                                 
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        self.mask_turn_list = []
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)
        self.overlap_wedge1.ori = -(fly_frame_ori - self.overlap_angle)%360
        self.overlap_wedge2.ori = -(fly_frame_ori + self.overlap_angle)%360


    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        min_angle = 360//len(self.mask_library_list[0])
        for k in range(self.n_segments):
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge1.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge2.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge3.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge4.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((0 * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*0)
                                                                                                                        + (self.translation_direction[k]*0))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask tracking the fly
            if flip:
                self.mask_turn_list[k] += 180
            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*0 + self.translation_direction[k]*0) + (0 * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+0)%360, min_angle)]
        self.overlap_wedge1.ori = -(fly_frame_ori - self.overlap_angle + 2)%360
        self.overlap_wedge2.ori = -(fly_frame_ori + self.overlap_angle - 2)%360
        self.overlap_wedge3.ori = -(fly_frame_ori - self.overlap_angle + 2)%360
        self.overlap_wedge4.ori = -(fly_frame_ori + self.overlap_angle - 2)%360
        for k in range(self.n_segments):
            self.stimulus[k].draw()
        self.overlap_wedge1.draw()
        self.overlap_wedge2.draw()
        self.overlap_wedge3.draw()
        self.overlap_wedge4.draw()
        return [0, 0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters=dict(direction=0), frame_number=0, flip=False):
        translation_angle=0
        min_angle = 360//len(self.mask_library_list[0])
        offset_translation = 3
        self.direction = list(trial_parameters['direction'])
        for k in range(self.n_segments):
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge1.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge2.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge3.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.overlap_wedge4.pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((self.direction[k] * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*translation_angle)
                                                                                                + (self.translation_direction[k]*offset_translation))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
            if flip:
                self.mask_turn_list[k] += 180
            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*translation_angle + self.translation_direction[k]*offset_translation) + (self.direction[k] * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+self.point_of_expansion[0])%360, min_angle)]
        self.overlap_wedge1.ori = -(fly_frame_ori - self.overlap_angle + 2)%360
        self.overlap_wedge2.ori = -(fly_frame_ori + self.overlap_angle - 2)%360
        self.overlap_wedge3.ori = -(fly_frame_ori - self.overlap_angle + 2)%360
        self.overlap_wedge4.ori = -(fly_frame_ori + self.overlap_angle - 2)%360
        for k in range(self.n_segments):
            self.stimulus[k].draw()
        self.overlap_wedge1.draw()
        self.overlap_wedge2.draw()
        self.overlap_wedge3.draw()
        self.overlap_wedge4.draw()
        return trial_parameters['direction']
    
    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('overlap_wedge1')
        features.pop('overlap_wedge2')
        features.pop('overlap_wedge3')
        features.pop('overlap_wedge4')
        features.pop('mask_library_list')
        features.pop('mask_turn_list')
        features.pop('stimulus_image')
        return features


class half_flicker(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        super().__init__(win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori)
        self.name = 'half_flicker'
        self.angle = []
        for k in range(self.n_segments):
            self.angle.append((360/(self.spatial_frequency[k] * 2)))
        
    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        translation_angle=0
        min_angle = 360//len(self.mask_library_list[0])
        offset_translation = 3
        self.direction = trial_parameters['direction']
        
        for k in range(self.n_segments):
            if frame_number % (2*self.temporal_frequency[k]) == 0:
                direction = self.direction[k] * 1
            elif frame_number % (self.temporal_frequency[k]) == 0:
                direction = self.direction[k] * -1
            else:
                direction = 0
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((direction * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*translation_angle)
                                                                                                                        + (self.translation_direction[k]*offset_translation))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*translation_angle + self.translation_direction[k]*offset_translation) + (direction * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+self.point_of_expansion[0])%360, min_angle)]
        for k in range(self.n_segments):
            self.stimulus[k].draw()
        return trial_parameters['direction']


class half_pinwheel_oscillate(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        super().__init__(win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori)
        self.name = 'half_flicker'
        self.angle = []
        for k in range(self.n_segments):
            self.angle.append((360/(self.spatial_frequency[k] * 2)))
        self.amplitude = 600
        self.speed = (((360/self.spatial_frequency[k]) * (self.temporal_frequency[k]))/framerate)
        
    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters=dict(direction=0), frame_number=0, flip=False):
        translation_angle=0
        min_angle = 360//len(self.mask_library_list[0])
        offset_translation = 3
        self.direction = list(trial_parameters['direction'])
        self.angle = self.amplitude - abs(self.speed * (frame_number % (self.amplitude/self.speed)))
        self.angle = [self.angle, self.angle]
        for k in range(self.n_segments):
            self.stimulus[k].pos = [super().transform_fly_pos(pos[0], pos[1])]
            self.stimulus[k].ori = (self.stimulus[k].ori + ((self.direction[k] * self.angle[k]) - (self.closedloop_gain_rotation[k]*fly_turn)) + ((self.closedloop_gain_translation[k]*translation_angle)
                                                                                                + (self.translation_direction[k]*offset_translation))) % 360
            # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
            if flip:
                self.mask_turn_list[k] += 180
            self.mask_turn_list[k] = (self.mask_turn_list[k] + (self.closedloop_gain_translation[k]*translation_angle + self.translation_direction[k]*offset_translation) + (self.direction[k] * self.angle[k])) % 360
            self.stimulus[k].mask = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+self.point_of_expansion[0])%360, min_angle)]
        for k in range(self.n_segments):
            self.stimulus[k].draw()
        return trial_parameters['direction']
    
    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('mask_library_list')
        features.pop('mask_turn_list')
        features.pop('stimulus_image')
        return features


class half_pinwheel_ring(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        self.ring_size = 25
        self.n_segments=n_segments
        self.contrast = contrast_list
        self.closedloop_gain_rotation = closedloop_gain_rotation_list
        self.temporal_frequency = temporal_frequency_list
        self.spatial_frequency = spatial_frequency_list
        self.closedloop_gain_translation = [0, 0]
        self.point_of_expansion = [0, 0]
        self.translation_direction = [0, 0]
        self.name = 'half_pinwheel_ring'
        self.win = win
        self.stimulus_image = []
        self.angle = []
        for k in range(self.n_segments):
            stimulus = 'ring_pinwheel'
            stimulus_filename = stimulus + '_' + str(int(self.spatial_frequency[k])) + '_' + str(int(self.ring_size)) +'.png'
            self.stimulus_image.append(os.path.join(destination, stimulus_filename))
            self.angle.append(((360/self.spatial_frequency[k]) * (self.temporal_frequency[k]))/framerate)


class quarter_pinwheel_old(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        segmented_pinwheel_old.__init__(win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list , fly_frame_ori)
        self.name = 'quarter_pinwheel'
    
    def create_stimulus(self):
        segmented_pinwheel_old.create_stimulus()

    def create_mask(self, fly_frame_ori):
        mask_library_ori = np.load('mask_data_90.npy')
        self.mask_library_list = []
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135+90+90)) #right
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135)) #left
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135+90)) #center
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=45))  #rear
                                 
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        self.mask_turn_list = []
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        return segmented_pinwheel_old.update_default(self, pos, fly_turn, fly_frame_ori, flip=flip)
    
    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        return segmented_pinwheel_old.update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=flip)

    def get_stimulus_features(self):
        return segmented_pinwheel_old.get_stimulus_features()


class quarter_pinwheel_old_front(segmented_pinwheel_old):
    def __init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list, fly_frame_ori):
        segmented_pinwheel_old.__init__(self, win, n_segments, contrast_list, spatial_frequency_list, temporal_frequency_list, framerate, closedloop_gain_rotation_list , fly_frame_ori)
        self.name = 'quarter_pinwheel_front'
    
    def create_stimulus(self):
        segmented_pinwheel_old.create_stimulus(self)

    def create_mask(self, fly_frame_ori):
        mask_library_ori = np.load('mask_data_90.npy')
        self.mask_library_list = []
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=135+90)) #center
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=45))  #rear
                                 
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        self.mask_turn_list = []
        self.mask_turn_list.append(fly_frame_ori)
        self.mask_turn_list.append(fly_frame_ori)

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        return segmented_pinwheel_old.update_default(self, pos, fly_turn, fly_frame_ori, flip=flip)
    
    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        return segmented_pinwheel_old.update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=flip)

    def get_stimulus_features(self):
        return segmented_pinwheel_old.get_stimulus_features(self)


class full_pinwheel_off_edge_clockwise(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        super().__init__(win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)
        self.name = 'full_pinwheel_off_edge'
        self.stimulus_image = os.path.join(destination, 'off_edge_clockwise' + '.png')


class full_pinwheel_on_edge_clockwise(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        super().__init__(win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)
        self.name = 'full_pinwheel_off_edge'
        self.stimulus_image = os.path.join(destination, 'on_edge_clockwise' + '.png')


class full_pinwheel_off_edge_counterclockwise(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        super().__init__(win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)
        self.name = 'full_pinwheel_off_edge'
        self.stimulus_image = os.path.join(destination, 'off_edge_clockwise' + '.png')


class full_pinwheel_on_edge_counterclockwise(full_pinwheel_old):
    def __init__(self, win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain):
        super().__init__(win, contrast, spatial_frequency, temporal_frequency, framerate, closedloop_gain)
        self.name = 'full_pinwheel_off_edge'
        self.stimulus_image = os.path.join(destination, 'on_edge_clockwise' + '.png')


class full_opposing_dots(transform_fly_pos):
    def __init__(self, win, N, coherence, contrast, stim_size, speed, framerate, rod, closedloop_gain):
        self.N = N
        self.n1 = int(self.N * coherence)
        self.n2 = int(self.N * (1 - coherence))
        dot_stim1 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n1, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        dot_stim2 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n2, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        self.stimulus = [dot_stim1, dot_stim2]
        self.name = 'full_opposing_dots'
        self.win = win
        self.stimulus[0].autoDraw = False
        self.stimulus[1].autoDraw = False
        self.contrast = contrast
        self.closedloop_gain = closedloop_gain
        self.speed = speed
        self.angle = speed
        self.framerate = framerate
        self.rod = rod
        theta = np.radians(speed)  # could use radians if you like
        self.rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        self.stimulus[0].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[1].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[0].draw()
        self.stimulus[1].draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.direction = trial_parameters['direction']
        self.stimulus[0].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[1].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        if self.rod==0:
            pass
        else:
            index1 = random.sample(range(self.n1), int(self.rod * self.n1))
            index2 = random.sample(range(self.n2), int(self.rod * self.n2))
            self.stimulus[0].xys[index1]=np.subtract(2*np.random.random((int(self.rod * self.n1), 2)), 1)
            self.stimulus[1].xys[index2]=np.subtract(2*np.random.random((int(self.rod * self.n2), 2)), 1)
        self.stimulus[0].xys = np.dot(self.stimulus[0].xys, self.rot_mat * [[1, trial_parameters['direction']], [trial_parameters['direction'], 1]])
        self.stimulus[1].xys = np.dot(self.stimulus[1].xys, self.rot_mat * [[1, -1*trial_parameters['direction']], [-1*trial_parameters['direction'], 1]])
        self.stimulus[0].draw()
        self.stimulus[1].draw()
        return [trial_parameters['direction']]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('rot_mat')
        return features


class full_random_dots(transform_fly_pos):
    def __init__(self, win, N, coherence, contrast, stim_size, speed, framerate, rod, closedloop_gain):
        self.N = N
        self.n1 = int(self.N * coherence)
        self.n2 = int(self.N * (1 - coherence))
        dot_stim1 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n1, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        dot_stim2 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n2, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        self.stimulus = [dot_stim1, dot_stim2]
        self.name = 'full_random_dots'
        self.win = win
        self.stimulus[0].autoDraw = False
        self.stimulus[1].autoDraw = False
        self.contrast = contrast
        self.closedloop_gain = closedloop_gain
        self.speed = speed
        self.angle = speed
        self.framerate = framerate
        self.rod = rod
        self.dotsize_data = np.load('dotsize_data.npy')

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        self.stimulus[0].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[1].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[0].draw()
        self.stimulus[1].draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.direction = trial_parameters['direction']
        self.stimulus[0].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[1].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        if self.rod==0:
            pass
        else:
            index1 = random.sample(range(self.n1), int(self.rod * self.n1))
            index2 = random.sample(range(self.n2), int(self.rod * self.n2))
            self.stimulus[0].xys[index1]=np.subtract(2*np.random.random((int(self.rod * self.n1), 2)), 1)
            self.stimulus[1].xys[index2]=np.subtract(2*np.random.random((int(self.rod * self.n2), 2)), 1)
        self.stimulus[0].xys = np.dot(self.stimulus[0].xys, self.rot_mat * [[1, trial_parameters['direction']], [trial_parameters['direction'], 1]])
        dots_lost_indices = np.argwhere(np.abs(self.stimulus[1].xys)>1)
        modifier = np.ones((int(self.n2), 2))
        modifier[dots_lost_indices, :] = -1

        self.translation_mat = self.translation_mat * modifier
        self.stimulus[1].xys = np.add(self.stimulus[1].xys, self.translation_mat)
        self.stimulus[0].draw()
        self.stimulus[1].draw()

        return [trial_parameters['direction']]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('rot_mat')
        features.pop('translation_mat')
        return features


class half_random_dots(transform_fly_pos):
    def __init__(self, win, N, coherence, contrast, stim_size, speed, framerate, rod, closedloop_gain):
        self.N = N
        self.n1 = int(self.N * coherence)//2
        self.n2 = int(self.N * coherence)//2
        dot_stim1 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n1, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        dot_stim2 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2.0, 2.0), fieldShape='circle', nElements=self.n2, sizes=stim_size,
                                                     xys=None, rgbs=None, colors=contrast, colorSpace='rgb',
                                                     opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                                     elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
        self.stimulus = [dot_stim1, dot_stim2]
        self.name = 'full_random_dots'
        self.win = win
        self.stimulus[0].autoDraw = False
        self.stimulus[1].autoDraw = False
        self.contrast = contrast
        self.closedloop_gain = closedloop_gain
        self.speed = speed
        self.angle = speed
        self.framerate = framerate
        self.rod = rod
        self.dotsize_data = np.load('dotsize_data.npy')
    
    def create_mask(self, fly_frame_ori):
        mask_library_ori = np.load('mask_data_180.npy')
        self.mask_library_list = []
        self.mask_library_list.append(np.array(mask_library_ori))
        self.mask_library_list.append(mask.create_inverse_mask(mask_library_ori, phase=180))
                                 
        # initialise mask_turn, it is the angle by which the whole stimulus (both halves) should turn in order to rotate with the fly
        self.mask_turn_list = []
        self.mask_turn_list.append(fly_frame_ori % 360)
        self.mask_turn_list.append(fly_frame_ori % 360)

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        min_angle = 360//len(self.mask_library_list[0])
        for k in range(2):
            ## find distance from center
            distance = np.round(np.sqrt(np.sum(np.square(np.subtract(self.stimulus[k].xys, [0, 0])), axis=1)), 2)
            # print(distance.shape, distance[0])
            # self.stimulus[k].sizes = np.round(distance, 3) * ((7.5/180) * 3.14)
            # dist_list = []
            # for i in distance:
            #     dist_list.append(self.dotsize_data[i==self.dotsize_data[:, 1], 0][0])
            # self.stimulus[k].sizes = dist_list
            ##
            rot_angle = -(self.closedloop_gain*fly_turn) % 360
            self.rot_mat = np.array([[np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))], [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]])
            self.stimulus[k].xys = np.dot(self.stimulus[k].xys, self.rot_mat)
            self.stimulus[k].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
            # mask_turn turns the mask by fly_turn, this results in the mask turning tracking the fly
            self.mask_turn_list[k] = fly_frame_ori % 360
            # convert xys to indices for mask matrix
            # indices = (np.add(self.stimulus[k].xys, 1)*64).astype(int)
            indices = (np.add(np.stack((self.stimulus[k].xys[:,1], self.stimulus[k].xys[:,0]), axis=1), 1)*64).astype(int)
            # find the mask value of all these indices
            mask_values = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k])%360, min_angle)]
            # choose the dots to "remove"
            indices_to_remove = np.argwhere(mask_values[indices[:, 0], indices[:, 1]] == -1).flatten()
            # set the opacties of those dots to 0
            opacities = np.ones(self.stimulus[k].xys.shape[0])
            opacities[indices_to_remove] = 0
            self.stimulus[k].opacities = opacities
        for k in range(2):
            self.stimulus[k].draw()
        return [0] * 2

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        point_of_expansion = 0
        min_angle = 360//len(self.mask_library_list[0])
        self.direction = list(trial_parameters['direction'])
        self.stimulus[0].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]
        self.stimulus[1].fieldPos = [super().transform_fly_pos(pos[0], pos[1])]

        if self.rod==0:
            pass
        else:
            index1 = random.sample(range(self.n1), int(self.rod * self.n1))
            index2 = random.sample(range(self.n2), int(self.rod * self.n2))
            # choose new dot locations such that its still in a circle, choose phi and theta and then convert to x and y
            phi = np.random.uniform(0, 2*np.pi, int(self.rod * self.n1))
            rad = np.sqrt(np.random.uniform(0, 9, int(self.rod * self.n2)))/3
            x = rad * np.cos(phi)
            y = rad * np.sin(phi)
            self.stimulus[0].xys[index1]=np.stack((x, y), axis=1)
            phi = np.random.uniform(0, 2*np.pi, int(self.rod * self.n2))
            rad = np.sqrt(np.random.uniform(0, 9, int(self.rod * self.n2)))/3
            x = rad * np.cos(phi)
            y = rad * np.sin(phi)
            self.stimulus[1].xys[index2]=np.stack((x, y), axis=1)

        for k in range(2):
            ## find distance from center
            distance = np.round(np.sqrt(np.sum(np.square(np.subtract(self.stimulus[k].xys, [0, 0])), axis=1)), 2)
            # print(distance.shape, distance[0])
            # self.stimulus[k].sizes = np.round(distance, 3) * (7.5/180) * 3.14
            # dist_list = []
            # for i in distance:
            #     dist_list.append(self.dotsize_data[i==self.dotsize_data[:, 1], 0][0])
            # self.stimulus[k].sizes = dist_list
            ##

            rot_angle = ((self.direction[k] * self.speed)-(self.closedloop_gain*fly_turn)) % 360
            self.rot_mat = np.array([[np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))], [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]])
            if self.direction[k] == 0:
                pass
            else:
                self.stimulus[k].xys = np.dot(self.stimulus[k].xys, self.rot_mat)
            # mask_turn turns the mask by fly_turn, this results in the mask turning tracking the fly
            self.mask_turn_list[k] = fly_frame_ori % 360
            # convert xys to indices for mask matrix
            # indices = (np.add(self.stimulus[k].xys, 1)*64).astype(int)
            indices = (np.add(np.stack((self.stimulus[k].xys[:,1], self.stimulus[k].xys[:,0]), axis=1), 1)*64).astype(int)
            # find the mask value of all these indices
            mask_values = self.mask_library_list[k][mask.angle_to_index((self.mask_turn_list[k]+point_of_expansion)%360, min_angle)]
            # choose the dots to "remove"
            indices_to_remove = np.argwhere(mask_values[indices[:, 0], indices[:, 1]] == -1).flatten()
            # set the opacties of those dots to 0
            opacities = np.ones(self.stimulus[k].xys.shape[0])
            opacities[indices_to_remove] = 0
            self.stimulus[k].opacities = opacities

        self.stimulus[0].draw()
        self.stimulus[1].draw()
        return trial_parameters['direction']

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        features.pop('rot_mat')
        features.pop('mask_library_list')
        features.pop('mask_turn_list')
        features.pop('dotsize_data')
        return features


class translation_grating(transform_fly_pos):
    def __init__(self, win, contrast, spatial, temporal):
        self.contrast = contrast
        self.win = win
        self.spatial_frequency = spatial
        self.temporal_frequency = temporal
        self.name = 'translation'
        grating_stim = psychopy.visual.GratingStim(
        win=win,
        units='norm',
        pos=(0, 0),
        size=(4, 4),
        tex='sqr',
        contrast=contrast,
        sf=spatial,
        ori=0,
        phase=0,
        )
        self.stimulus = grating_stim
        self.change_per_frame = 1/self.temporal_frequency

    def update_default(self, pos, fly_turn, fly_frame_ori, flip=False):
        self.stimulus.ori = fly_frame_ori
        transformed_pos = super().transform_fly_pos(pos[0], pos[1])
        self.stimulus.pos = [transformed_pos[0]+2*np.cos(fly_frame_ori), transformed_pos[1]+2*np.sin(fly_frame_ori)]
        self.stimulus.draw()
        return [0]

    def update_with_motion(self, pos, fly_turn, fly_frame_ori, trial_parameters, frame_number, flip=False):
        self.stimulus.ori = fly_frame_ori
        transformed_pos = super().transform_fly_pos(pos[0], pos[1])
        self.stimulus.pos = [transformed_pos[0]+2*np.cos(fly_frame_ori), transformed_pos[1]+2*np.sin(fly_frame_ori)]
        self.stimulus.phase += [self.angle*trial_parameters['direction'], self.angle*trial_parameters['direction']]
        self.stimulus.draw()
        return [trial_parameters['direction']]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        return features


class female_dot():
    def __init__(self, win, size, contrast, speed, amplitude, dist_female):
        self.contrast = contrast
        self.size = size
        dot_stim = psychopy.visual.Circle(
            win=win,
            units='norm',
            radius=1 / 15,
            pos=(0, 0),
            lineColor=-1,
            fillColor=-1
        )
        self.stimulus = dot_stim
        self.win = win
        self.stimulus.autoDraw = False
        self.speed = speed
        self.amplitude = amplitude
        self.dist_female = dist_female

class sinusoidal_dot(female_dot, transform_fly_pos):
    def __init__(self, win, size, contrast, speed, amplitude, dist_female):
        female_dot.__init__(self, win, size, contrast, speed, amplitude, dist_female)
        self.name = 'sinusoidal_dot'

    def update(self, pos, fly_frame_ori, trial_parameters=dict(direction=0), frame_number=0):
        angle = frame_number * 2 * self.speed % 360
        angle = self.amplitude * (np.sin((angle / 180) * np.pi))
        x, y = int(self.dist_female * np.cos((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][0]), int(
            self.dist_female * np.sin((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][1])
        x, y = transform_fly_pos.transform_fly_pos(x, y)
        self.stimulus.pos = [[x, y]]
        return [angle]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        return features


class triangular_dot(female_dot, transform_fly_pos):
    def __init__(self, win, size, contrast, speed, amplitude, dist_female):
        female_dot.__init__(self, win, size, contrast, speed, amplitude, dist_female)
        self.name = 'triangular_dot'

    def update(self, pos, fly_frame_ori, trial_parameters=dict(direction=0), frame_number=0):
        angle = self.amplitude - abs(2 * self.amplitude - (self.amplitude * frame_number % (4 * self.amplitude)))
        x_dot, y_dot = int(self.dist_female * np.cos((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][0]), int(
            self.dist_female * np.sin((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + pos[0][1])
        x_dot = (x_dot - 512) / w_ratio + w_rem
        y_dot = (y_dot - 512) / h_ratio + h_rem
        self.stimulus.pos = [[x_dot, y_dot]]
        return [angle]

    def get_stimulus_features(self):
        features = self.__dict__
        features.pop('win')
        features.pop('stimulus')
        return features


def initiliaze_stimulus(stimulus_type, win, stimulus_parameters, fly_ori):
    if stimulus_type == 'Full pinwheel':
        # stimulus = full_pinwheel_old(win, contrast=experiment_data['contrast'], spatial_frequency=experiment_data['spatial_frequency'],
        #                              temporal_frequency=experiment_data['temporal_frequency'], framerate=60, closedloop_gain=0)
        stimulus = full_pinwheel_old(win, contrast=stimulus_parameters['contrast'], spatial_frequency=stimulus_parameters['spatial_frequency'],
                                     temporal_frequency=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain=0)
        stimulus.create_stimulus()
    elif stimulus_type == 'Half pinwheel':
        # stimulus = full_pinwheel_old(win, contrast=experiment_data['contrast'], spatial_frequency=experiment_data['spatial_frequency'],
        #                              temporal_frequency=experiment_data['temporal_frequency'], framerate=60, closedloop_gain=0)
        stimulus = half_pinwheel_old(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'],
                                     temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'Full Random Dots':
        stimulus = full_random_dots(win, contrast=stimulus_parameters['contrast'], N=stimulus_parameters['N_dots'], coherence = stimulus_parameters['coherence'],
                                     rod=stimulus_parameters['rod'], framerate=60, closedloop_gain=0, speed=2, stim_size=0.1)
    elif stimulus_type == 'Full Opposing Dots':
        stimulus = full_opposing_dots(win, contrast=stimulus_parameters['contrast'], N=stimulus_parameters['N_dots'], coherence = stimulus_parameters['coherence'],
                                     rod=stimulus_parameters['rod'], framerate=60, closedloop_gain=0, speed=2, stim_size=0.1)
    elif stimulus_type == 'Half pinwheel flicker':
        stimulus = half_flicker(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'],
                    temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'Flicker pinwheel':
        stimulus = full_flicker(win, contrast=stimulus_parameters['contrast'], spatial_frequency=stimulus_parameters['spatial_frequency'],
                                     temporal_frequency=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain=0)
        stimulus.create_stimulus()
    elif stimulus_type == 'Half pinwheel ring':
        stimulus = half_pinwheel_ring(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'],
                    temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'Half pinwheel binocular overlap':
        stimulus = binocular_overlap(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'],
                    temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'Half random dots':
        stimulus = half_random_dots(win, contrast=-1, N=stimulus_parameters['N_dots'], coherence = 1,
                                     rod=stimulus_parameters['rod'], framerate=60, closedloop_gain=1, speed=5, stim_size=0.15)
        stimulus.create_mask(fly_ori)
    elif  stimulus_type == "Quarter pinwheel front":
        stimulus = quarter_pinwheel_old_front(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'], 
                                              temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, 
                                              closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'Half pinwheel oscillate':
        stimulus = half_pinwheel_oscillate(win, n_segments=2, contrast_list=stimulus_parameters['contrast'], spatial_frequency_list=stimulus_parameters['spatial_frequency'],
                    temporal_frequency_list=stimulus_parameters['temporal_frequency'], framerate=60, closedloop_gain_rotation_list=[1, 1], fly_frame_ori=fly_ori)
        stimulus.create_stimulus()
        stimulus.create_mask(fly_ori)
    elif stimulus_type == 'dark and light':
        stimulus = dark_and_light(win, contrast=stimulus_parameters['contrast'], framerate=60, closedloop_gain=0)
        stimulus.create_stimulus()
    elif stimulus_type == 'half_translation':
        stimulus = translation_grating(win, contrast=stimulus_parameters['contrast'], spatial=stimulus_parameters['spatial'], \
                                       temporal=stimulus_parameters['temporal'], framerate=60, closedloop_gain=0)
        stimulus.create_stimulus()
    return stimulus


def initialise_projector_window(Dir, pos=[2164, -28], size=[342, 684]):
    print('initialising projector window')
    win = psychopy.visual.Window(
        size=size,
        monitor=0,
        pos=pos,
        color=(0, 0, 0),
        fullscr=False,
        waitBlanking=True
        # winType='pyglet'
    )
    win.recordFrameIntervals = True
    win.refreshThreshold = 1 / 60 + 0.001
    win.saveFrameIntervals(fileName=os.path.join(Dir, 'file.log'))
    print('projector window initialised')
    return win


def get_stimulus_combinations(stimulus_type, n_trials, experiment_data):
    if stimulus_type == 'Full pinwheel' or stimulus_type == 'Flicker pinwheel':
        stimulus_variables_list = [experiment_data['contrast'], experiment_data['spatial_frequency'], experiment_data['temporal_frequency'], [1, -1]]
        n_stimulus_combinations = np.product([len(x) for x in stimulus_variables_list])
        stimulus_list = np.tile(np.array(np.meshgrid(*stimulus_variables_list)).T.reshape(-1, (len(stimulus_variables_list))),
                                [(n_trials // n_stimulus_combinations) + 1, 1])[:int(n_trials*n_stimulus_combinations)]
        rng = np.random.default_rng()
        stimulus_list = rng.permutation(stimulus_list, axis=0)
        trial_parameters = {'contrast':stimulus_list[:n_trials, 0], 'spatial_frequency':stimulus_list[:n_trials, 1], 'temporal_frequency':stimulus_list[:n_trials, 2], 
                            'direction':stimulus_list[:n_trials, 3]}
        trial_parameters = pd.DataFrame.from_dict(trial_parameters)
    elif stimulus_type == 'Half pinwheel' or stimulus_type == 'Half pinwheel ring' or stimulus_type == 'Half pinwheel flicker'\
        or stimulus_type == 'Half pinwheel binocular overlap' or stimulus_type == "Quarter pinwheel front" or stimulus_type == 'Half pinwheel oscillate' \
            or stimulus_type == 'half translation':
        stimulus_variables_list = [experiment_data['contrast'][0], experiment_data['contrast'][1], 
                                   experiment_data['spatial_frequency'][0], experiment_data['spatial_frequency'][1],
                                   experiment_data['temporal_frequency'][0], experiment_data['temporal_frequency'][1],
                                   [1, -1, 0], [1, -1, 0]]
        n_stimulus_combinations = np.product([len(x) for x in stimulus_variables_list])
        stimulus_list = np.tile(np.array(np.meshgrid(*stimulus_variables_list)).T.reshape(-1, (len(stimulus_variables_list))),
                                [(n_trials // n_stimulus_combinations) + 1, 1])[:int(n_trials*n_stimulus_combinations)]
        rng = np.random.default_rng()
        stimulus_list = rng.permutation(stimulus_list, axis=0)
        direction_list=stimulus_list[:, 6:]
        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==0),np.argwhere(direction_list[:, 1]==0))
        direction_list[indices[:indices.shape[0]//2]] = [1, 1]
        direction_list[indices[indices.shape[0]//2:]] = [-1, -1]

        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==1),np.argwhere(direction_list[:, 1]==-1))
        direction_list[indices[:indices.shape[0]//2]] = [0, 1]
        direction_list[indices[indices.shape[0]//2:]] = [1, 0]

        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==-1),np.argwhere(direction_list[:, 1]==1))
        direction_list[indices[:indices.shape[0]//2]] = [0, -1]
        direction_list[indices[indices.shape[0]//2:]] = [-1, 0]

        trial_parameters = {'contrast':list(stimulus_list[:n_trials, :2]), 'spatial_frequency':list(stimulus_list[:n_trials, 2:4]), 
                            'temporal_frequency':list(stimulus_list[:n_trials, 4:6]), 'direction':list(stimulus_list[:n_trials, 6:])}
        trial_parameters = pd.DataFrame.from_dict(trial_parameters)
    elif stimulus_type == "Full Random Dots" or stimulus_type == "Full Opposing Dots" or stimulus_type == "Half Opposing Dots":
        stimulus_variables_list = [experiment_data['contrast'], experiment_data['N_dots'], experiment_data['rod'], experiment_data['coherence'], [1, -1]]
        n_stimulus_combinations = np.product([len(x) for x in stimulus_variables_list])
        stimulus_list = np.tile(np.array(np.meshgrid(*stimulus_variables_list)).T.reshape(-1, (len(stimulus_variables_list))),
                                [(n_trials // n_stimulus_combinations) + 1, 1])[:int(n_trials*n_stimulus_combinations)]
        rng = np.random.default_rng()
        stimulus_list = rng.permutation(stimulus_list, axis=0)
        trial_parameters = {'contrast':stimulus_list[:n_trials, 0], 'N_dots':stimulus_list[:n_trials, 1], 'rod':stimulus_list[:n_trials, 2], 
                            'coherence':stimulus_list[:n_trials, 3], 'direction':stimulus_list[:n_trials, 4]}
        trial_parameters = pd.DataFrame.from_dict(trial_parameters)
    elif stimulus_type == 'Half random dots':
        stimulus_variables_list = [experiment_data['N_dots'], experiment_data['rod'], [1, -1, 0], [1, -1, 0]]
        n_stimulus_combinations = np.product([len(x) for x in stimulus_variables_list])
        stimulus_list = np.tile(np.array(np.meshgrid(*stimulus_variables_list)).T.reshape(-1, (len(stimulus_variables_list))),
                                [(n_trials // n_stimulus_combinations) + 1, 1])[:int(n_trials*n_stimulus_combinations)]
        rng = np.random.default_rng()
        stimulus_list = rng.permutation(stimulus_list, axis=0)
        direction_list=stimulus_list[:, 2:]
        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==0),np.argwhere(direction_list[:, 1]==0))
        direction_list[indices[:indices.shape[0]//2]] = [1, 1]
        direction_list[indices[indices.shape[0]//2:]] = [-1, -1]

        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==1),np.argwhere(direction_list[:, 1]==-1))
        direction_list[indices[:indices.shape[0]//2]] = [0, 1]
        direction_list[indices[indices.shape[0]//2:]] = [1, 0]

        indices = np.intersect1d(np.argwhere(direction_list[:, 0]==-1),np.argwhere(direction_list[:, 1]==1))
        direction_list[indices[:indices.shape[0]//2]] = [0, -1]
        direction_list[indices[indices.shape[0]//2:]] = [-1, 0]

        trial_parameters = {'N_dots':list(stimulus_list[:n_trials, 0]), 'rod':list(stimulus_list[:n_trials, 1]), 
                            'direction':list(stimulus_list[:n_trials,2:])}
        print('here')
        trial_parameters = pd.DataFrame.from_dict(trial_parameters)
    elif stimulus_type == 'dark and light':
        stimulus_variables_list = [experiment_data['contrast']]
        stimulus_list = np.tile(stimulus_variables_list, [(n_trials // 2) + 1])[:n_trials]
        rng = np.random.default_rng()
        stimulus_list = rng.permutation(stimulus_list, axis=0)
        stimulus_list = stimulus_list.T
        trial_parameters = {'contrast':list(stimulus_list[:n_trials, 0])}
        trial_parameters = pd.DataFrame.from_dict(trial_parameters)
    return trial_parameters

