import random
import psychopy.visual
import psychopy.event
import numpy as np
import numpy.ma as ma
import copy

x = 200
y = 200
win = psychopy.visual.Window(
    size=[500, 500],
    pos = [0,0],
    fullscr=False
)

n_dots = 50
adjust = [[0.02, 0]] * n_dots

dot_xys = [[0, 0]]
dot1_xys = [[0, 0]]
dot_size = [[0.01]]

print(dot_xys)

for dot in range(n_dots - 1):
    dot_x = random.uniform(-1, 1)
    dot_y = random.uniform(-1, 1)
    dot_xys.append([dot_x, dot_y])

for dot in range(n_dots - 1):
    dot_s = random.uniform(0.05, 0.2)
    dot_size.append([dot_s])

dot_stim = psychopy.visual.ElementArrayStim(
    win=win,
    units="norm",
    nElements=n_dots,
    elementTex=None,
    colors=(-1, -1, -1),
    elementMask="circle",
    xys=dot_xys,
    sizes=dot_size
)

clock = psychopy.core.Clock()
while(True):
    dot_stim.oris += 0.5
    dot_stim.draw()
    win.flip()
    if clock.getTime()>25:
        break;
    

win.close()
