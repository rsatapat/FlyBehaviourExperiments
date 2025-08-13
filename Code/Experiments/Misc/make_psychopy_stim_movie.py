import psychopy.visual
import skvideo.io as sk
import random
import numpy as np
import cv2
import mask

no_of_frames = 60*5
win = psychopy.visual.Window((200, 200),color = (1,1,1),fullscr=False)

Dir = r'D:\Roshan'
filename = 'hemi_stimulus4.mp4'
path = r'{}/{}'.format(Dir, filename)
# writer_small = sk.FFmpegWriter(filename + '.avi', inputdict={'-r': '60'}, outputdict={'-vcodec': 'libx264',
#                                                                                       '-crf': '0',
#                                                                                       '-preset': 'slow'})

################################
stimulus = 'sine_pinwheel'
bars = random.choice([4, 6])
contrast = random.choice([50])
Hz = random.choice([5])
stimulus_image_pinwheel = stimulus + '_' + str(contrast) + '_' + str(bars) + '.png'
stimulus_image_pinwheel = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image_pinwheel)
Dir = r'D:\Roshan\Project\PythonCodes\Stimulus'
stimulus_image_natural = r'{}/{}'.format(Dir, 'green-canopybw.jpg')
stimulus_image_random_dots = r'{}/{}'.format(r'D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', 'random_dots_50_10.png')

stimulus_image = random.choice([stimulus_image_pinwheel])
if stimulus_image == stimulus_image_pinwheel:
    stimulus_type = 0
elif stimulus_image == stimulus_image_natural:
    stimulus_type = 1
else:
    stimulus_type = 2

a = [1, -1]
direction = random.choice(a)
stim_size = 2
angle = (360 * Hz) / (60 * bars)
print(bars, angle, contrast)
# closedloop_gain = random.choice([0,0.25,0.5,0.75,1])
# closedloop_gain = random.choice([-1,-0.5,0,0.25,0.5,0.75,1,2])
closedloop_gain = 0

wheel_stim1 = psychopy.visual.ImageStim(
    win=win,
    image=stimulus_image_pinwheel,
    mask='circle',
    pos=(0, 0),
    ori=0,  # 0 is vertical,positive values are rotated clockwise
    size=2,
)
wheel_stim2 = psychopy.visual.ImageStim(
    win=win,
    image=stimulus_image_pinwheel,
    mask='circle',
    pos=(0, 0),
    ori=0,  # 0 is vertical,positive values are rotated clockwise
    size=2,
)
mask_library = np.load('mask_data_180.npy')
################################

N = 200
coherence = 1
n1 = int(N*coherence)
n2 = int(N*(1-coherence))

dot_stim1 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2, 2), fieldShape='circle', nElements=n1, sizes=0.05,
                                 xys=None, rgbs=None, colors=-1, colorSpace='rgb',
                                 opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                 elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)
dot_stim2 = psychopy.visual.ElementArrayStim(win, units='norm', fieldPos=(0.0, 0.0), fieldSize=(2, 2), fieldShape='circle', nElements=n2, sizes=0.1,
                                 xys=None, rgbs=None, colors=-1, colorSpace='rgb',
                                 opacities=1.0, depths=0, fieldDepth=0, oris=0, sfs=1.0, contrs=1, phases=0, elementTex='circle',
                                 elementMask='circle', texRes=48, interpolate=True, name=None, autoLog=None, maskParams=None)

angle = 1
theta = np.radians(angle)  # could use radians if you like
rot_mat1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
rot_mat2 = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
i = 0
image = []
closedloop_gain_rotation = 0
closedloop_gain_translation = 0
fly_turn = 0
translation_angle = 0
trans_direction =0
offset_translation=0
point_of_expansions = 0
mask_turn1 = 0
mask_turn2 = 0
mask_library_inverse = mask.create_inverse_mask(mask_library)
min_angle = 360//len(mask_library)
point_of_expansion = random.choice([0])
while(i < no_of_frames):
    # index1 = random.sample(range(n1), 2)
    # index2 = random.sample(range(n2), 2)
    # dot_stim2.xys[index2] = np.stack((np.subtract(2 * np.random.random_sample(2), 1), np.subtract(2 * np.random.random_sample(2), 1)), axis=1)
    # dot_stim1.xys[index1] = np.stack((np.subtract(2 * np.random.random_sample(2), 1), np.subtract(2 * np.random.random_sample(2), 1)), axis=1)
    # print(dot_stim.xys)
    # dot_stim1.xys = np.dot(dot_stim1.xys, rot_mat1)
    # dot_stim2.xys = np.dot(dot_stim2.xys, rot_mat2)
    # rdk_stim.dir = rdk_stim.dir + angle
    # rdk_stim.draw()
    # dot_stim1.draw()
    # dot_stim2.draw()
    dir1 = 1
    dir2 = 1
    angle1 = 2
    angle2 = 2
    # right half
    wheel_stim1.ori = (wheel_stim1.ori + ((dir1 * angle1) - (closedloop_gain_rotation * fly_turn)) + (
                (closedloop_gain_translation * translation_angle) + (trans_direction * offset_translation))) % 360
    # mask_turn turns the mask by all angles except the fly_turn, this results in the mask turning tracking the fly
    mask_turn1 = (mask_turn1 + (closedloop_gain_translation * translation_angle + trans_direction * offset_translation) + (dir1 * angle1)) % 360
    wheel_stim1.mask = mask_library[mask.angle_to_index((mask_turn1 + point_of_expansion) % 360, min_angle)]

    # left half
    wheel_stim2.ori = (wheel_stim2.ori + ((dir2 * angle2) - (closedloop_gain_rotation * fly_turn)) + (
                (closedloop_gain_translation * translation_angle) + (trans_direction * offset_translation))) % 360
    mask_turn2 = (mask_turn2 + (closedloop_gain_translation * translation_angle + trans_direction * offset_translation) + (dir2 * angle2)) % 360
    wheel_stim2.mask = mask_library_inverse[mask.angle_to_index((mask_turn2 + point_of_expansion) % 360, min_angle)]

    wheel_stim1.draw()
    wheel_stim2.draw()
    # win.flip()
    img = win.getMovieFrame(buffer="back")
    img = np.asarray(img)
    cv2.imshow('frame', img)
    cv2.waitKey(1)
    win.flip()
    i = i + 1

win.saveMovieFrames(path)
win.close()