#%%
import random
import time 
from pyquibbler import iquib, initialize_quibbler, q, quiby
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
initialize_quibbler()
# Initialize variables for FPS calculation
start_time = time.time()
frame_count = [0]  # Use a list to store frame_count as a mutable object

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 99), ylim=(-2, 2))
title = ax.text(0.5,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(frame):
    random.seed(frame)
    random_list = []

    for i in range(100):
        rand_float = random.uniform(-2, 2)
        random_list.append(rand_float)

    x = np.arange(0, 100, 1).tolist()
    y = random_list

    line.set_data(x, y)

    # Calculate FPS
    frame_count[0] += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count[0] / elapsed_time
    # print(f"frame_count: {frame_count[0]:d}")
    # print(f"current_time: {current_time:.2f}")
    # print(f"FPS: {fps:.2f}")
    title.set_text(f"FPS: {fps:.2f}")

    return line,title,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=15, blit=True)


# Show the figure
plt.show()