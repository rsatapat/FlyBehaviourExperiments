import numpy as np
import math
import cv2
import random
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

value_list = []
image = np.zeros((256, 256))
# def generate_glider(input):
#     output = list(np.ones(len(input)))
#     input = [-1 if i == 0 else 1 for i in input]
#     output[0] = random.choice([-1,1])
#     for i in range(1,len(output)):
#         if input[i-1]*output[i-1] == -1:
#             output[i] = 1
#         else:
#             output[i] = -1
#     output = [0 if i == -1 else 1 for i in output]
#     return output
size = 256
max_alpha = 127
for i in range(0, size):
    for j in range(0, size):
        x = i - size // 2
        y = size // 2 - j
        angle = cart2pol(x, y)[1]
        radius = cart2pol(x, y)[0]
        # if ((abs(angle)//(math.pi/4))-1)%2 == 0:
        if ((angle*4)//np.pi)%2 == 1:
            image[i,j] = (-np.cos((angle*4)) * max_alpha) + 127
        else:
            image[i, j] = (np.cos((angle * 4)) * max_alpha) + 127

cv2.imwrite('image_white_edge2.png', image)

# image = image.astype('uint8')
# cv2.imshow('frame', image)
# cv2.waitKey(1)
#
# image = (np.add(np.multiply(np.array(value_list), 255), 0))
# image = image.astype('uint8')
# cv2.imshow('frame1', image)
# cv2.waitKey(0)
