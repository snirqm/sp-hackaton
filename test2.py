#%%
from time import sleep
from pyquibbler import iquib, initialize_quibbler, q, quiby
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
initialize_quibbler()

# Figure setup:
plt.figure()
plt.axis('square')
plt.axis([-10, 10, -10, 10]);
b = iquib(1)
a = iquib(1)
x = np.linspace(-10, 10, 10000)
y = q(np.sin, a*x + b)

plt.plot(x, y)
# ax = plt.axes([0.3, 0.8, 0.3, 0.03])
# Slider(ax=ax, valmin=-10, valmax=10, valinit=a, valstep=1, label='a');
# ax = plt.axes([0.3, 0.7, 0.3, 0.03])
# Slider(ax=ax, valmin=-10, valmax=10, valinit=b, valstep=1, label='b');
plt.show()
# %%