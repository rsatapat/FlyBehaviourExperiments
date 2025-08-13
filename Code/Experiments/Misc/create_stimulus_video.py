import numpy as np
import psychopy.event
import psychopy.visual
import random
import time
import os
import mask
from PIL import Image
import skvideo.io as sk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##psychopy stimulus window initialisation
win = psychopy.visual.Window(
    size=[1024, 1024],
    color = (1,1,1),
    fullscr=False,
    waitBlanking = True
   # winType='pyglet'
)
win.recordFrameIntervals = True
win.refreshThreshold = 1 / 60 + 0.004

dot_stim = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1/15,
        pos=(0, 0),
        lineColor=-1,
        fillColor=-1
)
dot_stim.autoDraw = False

dot_stim2 = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1/15,
        pos=(0, 0),
        lineColor=-1,
        fillColor=-1
)
dot_stim.autoDraw = False

dot_stim3 = psychopy.visual.Circle(
        win=win,
        units='norm',
        radius=1/15,
        pos=(0, 0),
        lineColor=-1,
        fillColor=-1
)
dot_stim.autoDraw = False

bars = random.choice([6])
contrast = random.choice([25])
Hz = random.choice([5])
stimulus_image_pinwheel = 'pinwheel' + '_' + str(contrast) + '_' + str(bars) + '.png'
stimulus_image_pinwheel = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', 'large_sine_pinwheel100_6.png')
a = [1, -1]
direction = random.choice(a)
stim_size = 1
angle_opto = (360 * Hz) / (60 * bars)
print(bars, angle_opto)
wheel_stim = psychopy.visual.ImageStim(
    win=win,
    image=stimulus_image_pinwheel,
    mask='circle',
    pos=(0, 0),
    ori=0,  # 0 is vertical,positive values are rotated clockwise
    size=2,
)
wheel_stim.autoDraw = False

w_ratio = 512
w_rem = 0
h_ratio = 512
h_rem = 0

closedloop_gain = 0
frame_number = 0
fly_ori = 0
frame_lost = 0
dist_female = 300
dist_female2 = 400
speed_of_female = 4
angle = -100
x, y = -1, -1
stim = 0
# dot_stim.fillColor = random.choice([-1, -0.5, -0.2, 1])
# dot_stim.radius = random.choice([1/5, 1/10, 1/15, 1/20, 1/25])
dot_stim.fillColor = random.choice([-1])
dot_stim.lineColor = dot_stim.fillColor

rect_bright_stim = psychopy.visual.Rect(
    win=win,
    width = 2,
    height = 2,
    fillColor = 1,
    lineColor = 1,
    units = 'norm'
)
rect_bright_stim.autoDraw = False

rect_dark_stim = psychopy.visual.Rect(
    win=win,
    width = 2,
    height = 2,
    fillColor = -1,
    lineColor = -1,
    units = 'norm'
)
rect_dark_stim.autoDraw = False

grating = psychopy.visual.GratingStim(win, tex='sqr', mask='none', units='norm', pos=(0.0, 0.0), size=2, sf=2, ori=0.0,
                                      phase=(0.0, 0.0), texRes=128, color=(1.0, 1.0, 1.0), colorSpace='rgb',
                                      contrast=1.0, depth=0, interpolate=False, blendmode='avg',
                                      name=None, autoLog=None, autoDraw=False, maskParams=None)

wheel_stim = psychopy.visual.ImageStim(
    win=win,
    image=stimulus_image_pinwheel,
    mask='circle',
    pos=(0, 0),
    ori=0,  # 0 is vertical,positive values are rotated clockwise
    size=2,
)
wheel_stim.autoDraw = False

## to make hemi stimulus
a = [1, -1]
direction = random.choice(a)
stim_size = 1.5  # random.choice(size_stim)
closedloop_gain_rotation = random.choice([1])
closedloop_gain_translation = random.choice([0])

offset_translation = 3
bars1 = random.choice([6])
Hz1 = random.choice([5])
contrast1 = random.choice([100])
angle1 = (360 * Hz1) / (60 * bars1)

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
stimulus_image1 = 'large_sine_pinwheel100_6' + '.png'
stimulus_image1 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image1)

stimulus_image2 = 'large_sine_pinwheel100_6' + '.png'
stimulus_image2 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image2)

stimulus_image3 = 'large_sine_pinwheel100_6' + '.png'
stimulus_image3 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image3)

stimulus_image4 = 'large_sine_pinwheel100_6' + '.png'
stimulus_image4 = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image4)

mask_library_ori = np.load('mask_data_180.npy')
stim_size=2
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
mask_turn_list = []
fly_frame_ori_last = 0
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
syn_or_anti = random.choice([1, -1])
no_of_sections = 2
frames_lost = 0
##
stimulus_side = 'right_left'
i=0
# writer = sk.FFmpegWriter(os.path.join(r'D:\Roshan\Stimulus', 'hemi_video.mp4'), inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264', '-b': '30000000000', '-pix_fmt': 'yuvj420p'})
while(i<360):
    # pos = [[512,512]]
    # if i < 360:
    #     if 60 > i >= 0 or 180 > i >= 120 or 300 > i >= 240:
    #         print('here')
    #         rect_dark_stim.draw()
    #     elif 120 > i >= 60 or 240 > i >= 180 or 300 > i >= 240:
    #         rect_bright_stim.draw()
    # elif i == 360:
    #     time.sleep(0.5)
    #     grating.ori = grating.ori + 90
    # elif i >360 and i < 560:
    #     grating.phase = grating.phase + 0.05
    #     grating.draw()
    # elif i == 560:
    #     time.sleep(0.5)
    #     grating.ori = grating.ori + 90
    # elif i >= 560 and i < 760:
    #     grating.phase = grating.phase + 0.05
    #     grating.draw()
    # elif i == 760:
    #     time.sleep(0.5)
    #     grating.ori = grating.ori + 90
    # elif i >= 760 and i < 960:
    #     grating.phase = grating.phase + 0.05
    #     grating.draw()
    # elif i == 960:
    #     time.sleep(0.5)
    #     grating.ori = grating.ori + 90
    # elif i >= 960 and i < 1160:
    #     grating.phase = grating.phase + 0.05
    #     grating.draw()
    #############################
    # fly_frame_ori = 0
    # angle = frame_number * 2 * speed_of_female % 360
    # angle = 45 * (np.sin((angle / 180) * np.pi))
    # # dot_stim.radius = (angle/3)/512 + 1/15
    # x, y = int(dist_female * np.cos((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + 0), int(
    #     dist_female * np.sin((((fly_frame_ori + angle) / 180) * np.pi) + np.pi / 2) + 0)
    # x = (x - 512) / w_ratio + w_rem
    # y = (y - 512) / h_ratio + h_rem
    # dot_stim.pos = [[x, y]]
    # dot_stim.draw()
    ############################
    pos = [[512, 512]]
    x = (pos[0][0] - 512) / w_ratio + w_rem
    y = (pos[0][1] - 512) / h_ratio + h_rem
    wheel_stim1.pos = [[x, y]]
    wheel_stim2.pos = [[x, y]]
    wheel_stim3.pos = [[x, y]]
    wheel_stim4.pos = [[x, y]]
    translation_angle = 0

    dir=-1
    syn_or_anti = 1
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
    fly_turn = 0
    for k in range(no_of_sections):
        wheel_stim_list[k].ori = (wheel_stim_list[k].ori + ((dir_list[k] * angle_list[k]) - (closedloop_gain_rotation * fly_turn)) + ((closedloop_gain_translation * translation_angle)
                    + (trans_direction * offset_translation))) % 360
        # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
        mask_turn_list[k] = (mask_turn_list[k] + (closedloop_gain_translation * translation_angle + trans_direction * offset_translation) + (dir_list[k] * angle_list[k])) % 360
        wheel_stim_list[k].mask = mask_library_list[k][mask.angle_to_index((mask_turn_list[k] + point_of_expansion) % 360, min_angle)]

    for k in range(no_of_sections):
        wheel_stim_list[k].draw()
    win.flip()
    ############################
    # dir = 1
    # if 0<= i < 360:
    #     wheel_stim.ori = angle
    #     wheel_stim.draw()
    # elif 300<= i < 600:
    #     wheel_stim.ori = wheel_stim.ori + dir * 5
    #     wheel_stim.draw()
    # elif 600<= i < 900:
    #     wheel_stim.ori = wheel_stim.ori +0
    #     wheel_stim.draw()
    # elif 900<=i < 1200:
    #     wheel_stim.ori = wheel_stim.ori + -1 * dir * 5
    #     wheel_stim.draw()

    # wheel_stim.ori = wheel_stim.ori + angle_opto
    # wheel_stim.draw()
    #############################
    image = win.getMovieFrame()
    # writer.writeFrame(np.asarray(image))
    win.flip()
    i = i + 1
    frame_number = frame_number + 1
Dir = r'D:\Roshan\Stimulus'
win.saveMovieFrames(fileName=os.path.join(Dir, 'hemistimulus_'+stimulus_side+'_FTB.mp4'))
# writer.close()

## animation of angular speed and cumulative turns
fig = plt.figure()
data = np.random.random(600)
start = 180
x = np.arange(start, data.shape[0])
y = data[start:]
ims = []
print(x.shape)
for i in range(x.shape[0]):
    ims.append(plt.plot(x[:i], y[:i], c='k'))

im_ani = animation.ArtistAnimation(fig, ims, repeat_delay=0, blit=True)
Dir = r'D:\Roshan\Experiments\Optomotor\NewDataSmallArena\FinalData\Figures'
im_ani.save(os.path.join(Dir, 'cumulative_response.mp4'), writer='ffmpeg', fps=60)