import numpy as np
import copy
import math
import cv2
import psychopy.visual
import misc

def cart2pol(x, y):
    '''
    input -> output
    tuple(int)[x-y coordinates] -> tuple(float)[polar radius, polar angle]
    '''
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)  # correct for quadrants [-135,-45,45,135] [return in radians, NOT degrees]
    return (rho, phi)


def cart2pol2(x, y):
    '''
    input -> output
    tuple(int)[x-y coordinates] -> tuple(float)[polar radius, polar angle]
    '''
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)  # correct for quadrants [-135,-45,45,135] [return in radians, NOT degrees]
    if phi < 0:
        phi = 2 * math.pi + phi
    return (rho, phi)


def create_circle_mask(num, segment):
    '''
    :param num: radius of mask in pixels
    :param flipped: mirror reflection
    :return: numpy array of circular masks
    '''
    circle_mask = np.negative(np.ones((num,num)))
    segment_radians = (segment / 180) * math.pi
    for i in range(num):
        for j in range(num):
            i1 = num // 2 - i
            j1 = j - num // 2
            angle_to_compare = cart2pol2(i1, j1)[1]
            if angle_to_compare > 0 and angle_to_compare < segment_radians:
                circle_mask[i][j] = 1

    for i in range(num):
        for j in range(num):
            if cart2pol(i - num // 2, j - num // 2)[0] > num // 2:
                circle_mask[i][j] = -1
    return circle_mask


def create_turning_mask(mask, num, segment, angle=18, flipped=False):
    turn = [-1, 1]
    unit_turn = (angle / 180) * math.pi
    mask_library = []
    offset = (segment / 180) * math.pi
    for k in range(360 // int(angle)):
        for i in range(num):
            for j in range(num):
                i1 = num // 2 - i
                j1 = j - num // 2
                if cart2pol2(i1, j1)[0] <= num // 2:
                    if cart2pol2(i1, j1)[1] >= k * unit_turn and cart2pol2(i1, j1)[1] < (k + 1) * unit_turn:
                        mask[i][j] = turn[0]
                    elif cart2pol2(i1, j1)[1] >= (k * unit_turn + offset) % (2 * math.pi) and cart2pol2(i1, j1)[
                        1] < ((k + 1) * unit_turn + offset) % (2 * math.pi):
                        mask[i][j] = turn[1]
                else:
                    pass
        new_mask = copy.deepcopy(mask)
        mask_library.append(new_mask)
    return mask_library


def create_360_turning_mask(num, angle, segment):
    new_circle_mask = create_circle_mask(num, segment)
    turning_masks = create_turning_mask(new_circle_mask, num, segment, angle)
    return turning_masks


def angle_to_index(angle, size_of_turn):
    '''
    :param angle: angle at which the stimulus needs to be oriented
    :param size_of_turn: minimum turn that the stimulus makes
    :return: index that is used to access one value in list : mask_library
    '''
    return int(angle // size_of_turn)


def create_reverse_mask(mask_library):
    mask_library_reverse = mask_library[::-1]
    mask_library_reverse = np.insert(mask_library_reverse, 0, mask_library_reverse[len(mask_library_reverse) - 1],
                                     axis=0)
    mask_library_reverse = mask_library_reverse[0:len(mask_library_reverse) - 1]
    return mask_library_reverse


def create_inverse_mask(mask_library, phase=180):
    mask_library_inverse = np.zeros(len(mask_library), dtype=object)
    print(mask_library_inverse)
    for i in range(len(mask_library)):
        for j in range(len(mask_library)):
            if abs(i - j) == len(mask_library) // (360//phase):
                mask_library_inverse[i] = mask_library[j]
    print(mask_library_inverse[0].shape)
    return mask_library_inverse


if __name__ == '__main__':
    angle = 1
    segment = 120
    mask_library = create_360_turning_mask(8, angle, segment)
    np.save('mask_data_new'+str(segment)+'.npy', mask_library)
    mask_library_inverse = create_inverse_mask(mask_library)

    win = misc.initialise_projector_window(r'D:\Roshan\Project\PythonCodes\Codes\Experiments')
    stimulus = 'sine_pinwheel_new127'
    stimulus_image_pinwheel = stimulus + '_' + '100' + '_' + '6' + '.png'
    stimulus_image_pinwheel = r'{}\{}'.format('D:\Roshan\Project\PythonCodes\Stimulus\Generate stimulus', stimulus_image_pinwheel)
    wheel_stim = psychopy.visual.ImageStim(
        win=win,
        image=stimulus_image_pinwheel,
        mask='circle',
        pos=(0, 0),
        ori=-9,  # 0 is vertical,positive values are rotated clockwise
        size=1,
        )
    wheel_stim.autoDraw = True

    k = -1
    while (True):
        dir = 1
        wheel_stim.ori = wheel_stim.ori + dir * 18
        wheel_stim.mask = mask_library[k]
        wheel_stim.draw()
        k = (k + 1) % int((360//angle))
        win.flip()
        cv2.waitKey(200)
        cv2.namedWindow('frame')  # Create a named window
        cv2.moveWindow('frame', 400, 300)  # Move it to (40,30)
        for i in range(len(mask_library)):
            image = mask_library_inverse[i] * 255
            cv2.imshow('frame', image)
            cv2.waitKey(20)

