import random
import numpy as np
import numpy.ma as ma
import copy
import cv2
import math
import cv2


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def contrast_to_pixel(contrast,max_pixel_value = 255):
    contrast = contrast / 100
    if contrast > 1:
        contrast == 1
    elif contrast < 0:
        contrast == 0

    max_alpha = int((contrast + 1) * max_pixel_value / 2)
    min_alpha = max_pixel_value - max_alpha

    return max_alpha,min_alpha

def pinwheel(size, bars,image = [], contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha,min_alpha)          # alpha == 0 means 100% transparency
    image = np.tile([max_alpha,min_alpha,min_alpha,255],(size,size,1))
    pix_val = [min_alpha,min_alpha,min_alpha,255]
    pix_val2 = [0,0,0,0]
    print(pix_val)
    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            radius = cart2pol(x, y)[0]
            # if angle < 0:
            #     image[i, j] = pix_val
            for n in range(1, bars, 2):
                if radius > 0:
                    if angle > n * math.pi / bars and angle < (n + 1) * math.pi / bars:
                        image[i, j] = pix_val
                    elif angle < -1*(n-1) * math.pi / bars and angle > -1*n * math.pi / bars:
                        image[i, j] = pix_val
                    else:
                        pass
                # else:
                #     image[i, j] = pix_val2
    return image

def ring_sine_pinwheel(size, bars, contrast = 100, ring_size=0.5):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha,min_alpha)          # alpha == 0 means 100% transparency
    image = np.tile([0, 0, 0, 0], (size, size, 1))
    pix_val = [min_alpha,min_alpha,min_alpha]
    pix_val2 = [0,0,0]
    print(pix_val)
    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            radius = cart2pol(x, y)[0]
            # if angle < 0:
            #     image[i, j] = pix_val
            for n in range(1, bars, 2):
                if radius > int(size*ring_size*0.5):
                    value = (math.sin(angle*bars) * (max_alpha-min_alpha)) + min_alpha ## wrong
                    # value = (((math.sin(angle * bars) + 1) / 2) * (max_alpha - min_alpha)) + min_alpha ## right
                    image[i, j] = [value, value, value, 255]
                    # if angle > n * math.pi / bars and angle < (n + 1) * math.pi / bars:
                    #     image[i, j] = pix_val
                    # elif angle < -1*(n-1) * math.pi / bars and angle > -1*n * math.pi / bars:
                    #     image[i, j] = pix_val
                    # else:
                    #     pass
                # else:
                #     image[i, j] = pix_val2
    return image

def sine_pinwheel(size, bars,image = [], contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast, max_pixel_value=127)
    print(max_alpha,min_alpha)          # alpha == 0 means 100% transparency
    image = np.tile([max_alpha,max_alpha,max_alpha],(size,size,1))
    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            radius = cart2pol(x, y)[0]
            # if angle < 0:
            #     image[i, j] = pix_val
            for n in range(1, bars, 2):
                # if size//2 > radius > 0:
                ## +1 to make the sinusoid positive, mulitply to change range
                    # value = (math.sin(angle*bars) * (max_alpha-min_alpha)) + min_alpha ## wrong
                    value = (((math.sin(angle * bars) + 1) /2) * (max_alpha-min_alpha)) + min_alpha ##right
                    image[i, j] = [value, value, value]
                    # if angle > n * math.pi / bars and angle < (n + 1) * math.pi / bars:
                    #     image[i, j] = pix_val
                    # elif angle < -1*(n-1) * math.pi / bars and angle > -1*n * math.pi / bars:
                    #     image[i, j] = pix_val
                    # else:
                    #     pass
                # else:
                #     image[i, j ] = [255, 255, 255, 0]
    return image

def pinwheel_edge(size=256, direction=1, type='sine', bars=8):
    value_list = []
    image = np.zeros((size, size))
    size = size
    max_alpha = 127
    if type=='sine':
        for i in range(0, size):
            for j in range(0, size):
                x = i - size // 2
                y = size // 2 - j
                angle = cart2pol(x, y)[1]
                radius = cart2pol(x, y)[0]
                if ((angle * (bars//2)) // np.pi) % 2 == 1: # tricky to explain :(, but it works
                    image[i, j] = (-np.cos((angle * (bars//2))) * max_alpha * direction) + 127
                else:
                    image[i, j] = (np.cos((angle * (bars//2))) * max_alpha * direction) + 127
    elif type=='linear':
        for i in range(0, size):
            for j in range(0, size):
                x = i - size // 2
                y = size // 2 - j
                angle = cart2pol(x, y)[1]
                radius = cart2pol(x, y)[0]
                if direction==-1:
                    image[i, j] = ((angle%(2*np.pi/bars)) * max_alpha * 2)
                elif direction==1:
                    image[i, j] = 255 - ((angle%(2*np.pi/bars)) * max_alpha * 2)
    return image

def pinwheel_dots(size, bars, image = [], contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha, min_alpha)  # alpha == 0 means 100% transparency
    image = np.tile([max_alpha,max_alpha,max_alpha,255],(size,size,1))
    a = [1,-1]
    radius = 0
    prob = 0.04
    for i in range(0,size):
        for j in range(0,size):
            x = i - size//2
            y = size//2 - j
            val = cart2pol(x,y)
            angle = val[1]
            rad = val[0]
            if rad > radius:
                for n in range(1, bars, 2):
                    points = []
                    if angle > n * math.pi / bars and angle < (n + 1) * math.pi / bars:
                        image[i,j] = [0, 0, 0, 0]
                        b=np.random.choice(a,p=[prob, 1-prob])
                        if b == 1:
                            rad = np.random.choice(5,1)
                            cv2.circle(image,(i,j),rad,[0,0,0,255],-1)
                    elif angle < -1 * (n - 1) * math.pi / bars and angle > -1 * n * math.pi / bars:
                        image[i, j] = [0, 0, 0, 0]
                        b = np.random.choice(a, p=[prob, 1 - prob])
                        if b == 1:
                            rad = np.random.choice(5, 1)
                            cv2.circle(image, (i, j), rad, [0, 0, 0, 255], -1)
    return image

def pinwheel_bar(size, bars, contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha, min_alpha)  # alpha == 0 means 100% transparency

    image = np.tile([max_alpha, max_alpha, max_alpha, 0], (size, size, 1))
    pix_val = [min_alpha, min_alpha, min_alpha, 255]

    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            if angle < 0:
                angle = angle + 2*math.pi
            print(angle)
            if angle > 4*(math.pi/4) - (math.pi / bars)/2 and angle < 4*(math.pi/4) + (math.pi / bars)/2:
                image[i, j] = pix_val
    return image

def pinwheel_bar_transparent(size, bars,image = [], contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha,min_alpha)          # alpha == 0 means 100% transparency
    image = np.tile([0, 0, 0, 127],(size,size,1))
    pix_val = [0, 0, 0, min_alpha]
    # pix_val2 = [0,0,0,0]
    print(pix_val)
    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            radius = cart2pol(x, y)[0]
            for n in range(1, bars, 2):
                if radius > 0:
                    if angle > n * math.pi / bars and angle < (n + 1) * math.pi / bars:
                        image[i, j] = pix_val
                    elif angle < -1*(n-1) * math.pi / bars and angle > -1*n * math.pi / bars:
                        image[i, j] = pix_val
                    else:
                        pass
    return image

def sine_pinwheel_transparent(size, bars,image = [], contrast = 100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha,min_alpha)          # alpha == 0 means 100% transparency
    image = np.tile([0, 0, 0, 127],(size,size,1))
    pix_val = [0, 0, 0, 127]
    print(pix_val)
    for i in range(0, size):
        for j in range(0, size):
            x = i - size // 2
            y = size // 2 - j
            angle = cart2pol(x, y)[1]
            radius = cart2pol(x, y)[0]
            # if angle < 0:
            #     image[i, j] = pix_val
            for n in range(1, bars, 2):
                if radius > 0:
                    value = (((math.sin(angle*bars)+1)/2) * (max_alpha-min_alpha)) + min_alpha
                    image[i, j ] = [0, 0,0, value]
    return image

def random_dots(size, density, dot_size = [1,5], contrast=100):
    max_alpha, min_alpha = contrast_to_pixel(contrast)
    print(max_alpha, min_alpha)  # alpha == 0 means 100% transparency

    image = np.tile([max_alpha, max_alpha, max_alpha, 255], (size, size, 1))
    num_of_dots = (density*size*size)/(np.pi*((dot_size[0]+dot_size[1])/2)**2)
    dots = 0
    while(dots < num_of_dots):
        pos_x = random.choice(range(0,size))
        pos_y = random.choice(range(0, size))
        
        # for dots with random radii
        radius = random.choice(range(dot_size[0], dot_size[1]))

        # for dots with radii based on distance from center
        # dist = math.sqrt((pos_x-256)**2 + (pos_y-256)**2)//50
        # radius = int(dist+ 1)
        # print(pos_x,pos_y,radius)
        image = cv2.circle(image, (pos_x,pos_y),radius, color=[min_alpha, min_alpha, min_alpha, 255], thickness=-1)
        dots = dots + 1
    return image

if __name__ == '__main__':
    size = 512
    bars = 6

    image = random_dots(size, 0.25)
    cv2.imwrite('random_dots_{}_{}.png'.format(50,10), image)

    # for bars in [4,6,12]:
    #     for contrast in [0,10,25,50,75,100]:
    #         cv2.imwrite('pinwheel_'+str(contrast)+'_'+str(bars)+'.png',pinwheel(size,bars,contrast=contrast))

    # for bars in [6]:
    #     for contrast in [100]:
    #         cv2.imwrite('pinwheel_dot_'+str(contrast)+'_'+str(bars)+'.png',pinwheel_dots(size,bars,contrast=contrast))

    # for bars in [6]:
    #     for contrast in [100]:
    #         cv2.imwrite('pinwheel16_blue' + str(contrast) + '_' + str(bars) + '.png',
    #                     pinwheel(size, bars, contrast=contrast))
# prob = int(1/prob)
# cv2.imwrite('random_dot_pinwheel'+str(prob)+'.png',pinwheel_dots(image))


