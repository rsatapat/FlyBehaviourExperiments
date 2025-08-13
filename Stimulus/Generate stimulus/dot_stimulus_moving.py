import random
import psychopy.visual
import psychopy.event
import numpy as np
import numpy.ma as ma
import copy

x = 200
y = 200
win = psychopy.visual.Window(
    size=[342, 684],
    pos = [1991,-20],
    #color = (0,0,0),
    units="norm",
    fullscr=False
)

'''
creates random dots and then the dots move across the roof 
once the dots are gone from the field of view, new dots are created
'''

n_dots = 50
adjust = [[0.02,0]]*n_dots

dot_xys = [[0,0]]
dot1_xys = [[0,0]]
dot_size = [[0.01]]
        
print(dot_xys)


for dot in range(n_dots-1):
    dot_x = random.uniform(-1, 1)
    dot_y = random.uniform(-1, 1)
    dot_xys.append([dot_x, dot_y])

for dot in range(n_dots-1):
    dot_s = random.uniform(0.05,0.2)
    dot_size.append([dot_s])

dot_stim = psychopy.visual.ElementArrayStim(
    win=win,
    units="norm",
    nElements=n_dots,
    elementTex=None,
    colors = (-1,-1,-1),
    elementMask="circle",
    xys=dot_xys,
    sizes=dot_size
)

clock = psychopy.core.Clock()
while(True):
    dot_stim.xys = dot_stim.xys - adjust        # fixed value subtracted from the location of all dots
    pos = copy.deepcopy(dot_stim.xys)
    if np.min(pos)<-1:
        this = np.argwhere(pos == np.min(pos))[0][0]    # x coordinate of the dot that is soon going to disappear due to motion
        pos = np.delete(pos,this,0)                     # ^this dot is deleted
        add = [[1,random.uniform(-1,1)]]                # another dot is added in its^ position
        pos = np.insert(pos,this,add,axis=0)
    dot_stim.xys = pos
    dot_stim.draw()
    # wheel_stim.draw()
    # wheel_stim.ori += 2
    win.flip()
    if clock.getTime()>25:
        break;
    

win.close()
