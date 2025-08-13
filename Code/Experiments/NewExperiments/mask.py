import numpy as np
import copy
import math
import cv2
import psychopy.visual


def cart2pol(x, y):
    '''
    input -> output
    tuple(int)[x-y coordinates] -> tuple(float)[polar radius, polar angle]
    '''
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)              # correct for quadrants [-135,-45,45,135] [return in radians, NOT degrees]
    return (rho, phi)

def cart2pol2(x, y):
    '''
    input -> output
    tuple(int)[x-y coordinates] -> tuple(float)[polar radius, polar angle]
    '''
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)              # correct for quadrants [-135,-45,45,135] [return in radians, NOT degrees]
    phi = phi % (2*math.pi)
    # if phi < 0:
    #     phi = 2*math.pi + phi
    return (rho, phi)

def create_circle_mask(num, segment = 180, flipped=False):
    '''
    makes 2d array of the shape of the segment
    :param num: size of the mask
    :param segment: angular size of the segment in angles
    :param flipped: whether the segment is flipped along the vertical axis
    :return:
    '''
    if segment == 180:
        if flipped == False:
            #circle_mask = np.concatenate((np.negative(np.ones((num, num//2))), np.ones((num, num//2))), axis = 1)
            circle_mask = np.concatenate((np.ones((num, num // 2)), (np.negative(np.ones((num, num // 2))))), axis=1)
        elif flipped == True:
            circle_mask = np.concatenate((np.ones((num, num // 2)),(np.negative(np.ones((num, num // 2))))), axis=1)
    # in projector coordinates, axis = 0 means bottom-up mask, axis = 1 means left+right mask

    else:
        segment=(segment/180)*math.pi
        circle_mask = np.negative(np.ones((num, num)))
        for i in range(num):
            for j in range(num):
                if cart2pol(i-num//2, j-num//2)[0] <= num//2 and cart2pol(i-num//2, j-num//2)[1]<=(segment-math.pi):
                    circle_mask[i][j] = 1
        print(segment)
        print(num // 2)
    return circle_mask


def create_turning_mask(mask, num, segment, angle=18, flipped=False):
    """
    rotates the mask by unit angle and creates a list of segment arrays
    :param mask: circular mask made by create_circle_mask
    :param num: size of mask
    :param segment: angular size of the segment in angles
    :param angle: unit angle by which the mask rotates through the list
    :param flipped:
    :return:
    """
    segment=(segment/180)*math.pi
    print(segment)
    if flipped:
        turn = [1, -1]
    else:
        turn = [-1, 1]

    if flipped == False:
        print(mask.shape)
        unit_turn = (angle/180) * math.pi

        mask_library = []
        for k in range(360//int(angle)):
            for i in range(num):
                for j in range(num):
                    i1 = num//2 - i
                    j1 = j - num//2
                    if cart2pol2(i1, j1)[0] <= num//2:
                        if ((k+1)*unit_turn + (2*math.pi -segment))==2*math.pi:
                            if cart2pol2(i1,j1)[1]>=k*unit_turn and cart2pol2(i1,j1)[1]<(k+1)*unit_turn:
                                mask[i][j] = turn[1]
                                # pass
                            elif cart2pol2(i1,j1)[1]>=(k*unit_turn + (2*math.pi -segment))%(2*math.pi) and cart2pol2(i1,j1)[1]<2*math.pi:
                                mask[i][j] = turn[0]
                        else:
                            if cart2pol2(i1,j1)[1]>=k*unit_turn and cart2pol2(i1,j1)[1]<(k+1)*unit_turn:
                                mask[i][j] = turn[1]
                                # pass
                            elif cart2pol2(i1,j1)[1]>=(k*unit_turn + (2*math.pi -segment))%(2*math.pi) and cart2pol2(i1,j1)[1]<((k+1)*unit_turn + (2*math.pi -segment))%(2*math.pi):
                                mask[i][j] = turn[0]
                            # pass
                    else:
                        pass
            new_mask = copy.deepcopy(mask)
            mask_library.append(new_mask)
    elif flipped == True:
        print(mask.shape)
        unit_turn = (angle / 180) * math.pi
        mask_library = []
        for k in range(180 // int(angle)):
            for i in range(num):
                for j in range(num):
                    i1 = num // 2 - i
                    j1 = j - num // 2
                    weird_condn = math.pi / 2 + (k + 1) * unit_turn
                    if weird_condn >= math.pi:
                        weird_condn = weird_condn - 2*math.pi
                    #print(weird_condn)
                    if cart2pol(i1, j1)[0] <= num // 2:
                        if cart2pol(i1, j1)[1] >= k * unit_turn and cart2pol(i1, j1)[1] < (k + 1) * unit_turn:
                            mask[i][j] = turn[1]
                        elif cart2pol(i1, j1)[1] >= weird_condn - unit_turn and cart2pol(i1, j1)[1] <  weird_condn:
                            mask[i][j] = turn[0]
                    else:
                        pass
            new_mask = copy.deepcopy(mask)
            mask_library.append(new_mask)
    return mask_library


def create_360_turning_mask(num, sector, angle):
    new_circle_mask = create_circle_mask(num, sector)
    turning_masks = create_turning_mask(new_circle_mask, num, sector, angle)
    return turning_masks

def angle_to_index(angle, size_of_turn):
    '''
    :param angle: angle at which the stimulus needs to be oriented
    :param size_of_turn: minimum turn that the stimulus makes
    :return: index that is used to access one value in list : mask_library
    '''
    return int(angle//size_of_turn)

def create_reverse_mask(mask_library):
    mask_library_reverse = mask_library[::-1]
    mask_library_reverse = np.insert(mask_library_reverse,0, mask_library_reverse[len(mask_library_reverse) - 1],axis = 0)
    mask_library_reverse = mask_library_reverse[0:len(mask_library_reverse) - 1]
    return mask_library_reverse

# def create_inverse_mask(mask_library, phase=180):
#     mask_library_inverse = np.zeros(len(mask_library), dtype=object)
#     for i in range(len(mask_library)):
#         for j in range(len(mask_library)):
#             if abs(i - j) == len(mask_library)//2:
#                 mask_library_inverse[i] = mask_library[j]
#     return mask_library_inverse

def create_inverse_mask(mask_library, phase=180):
    mask_library = np.array(mask_library)
    offset = int(len(mask_library)//(360/phase))
    mask_library_inverse = np.zeros(mask_library.shape)
    if phase >= 0:
        mask_library_inverse[offset:] = mask_library[:mask_library.shape[0]-offset]
        mask_library_inverse[:offset] = mask_library[mask_library.shape[0]-offset:]
    else:
        mask_library_inverse[offset:] = mask_library[mask_library.shape[0]+offset:]
        mask_library_inverse[:offset] = mask_library[:mask_library.shape[0]+offset]
    return mask_library_inverse

if __name__ == '__main__':
    angle = 1
    mask_library = create_360_turning_mask(128, 80, angle)
    np.save('mask_data_80.npy', mask_library)
    mask_library_inverse = np.array(mask_library)

    k = -1
    while(True):
        cv2.namedWindow('frame')
        cv2.moveWindow('frame', 400, 300)
        for i in range(len(mask_library)):
            image = mask_library_inverse[i]*255
            cv2.imshow('frame', image)
            cv2.waitKey(10)

