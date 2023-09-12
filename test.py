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
plt.axis([0, 10, 0, 10]);
b = iquib(1)
p = iquib(1)
x = p * np.sin(np.linspace(0, 10, 100))
y = p * np.cos(np.linspace(0, 10, 100))

# Generate a 2D grid of x and y values
X, Y = np.meshgrid(x, y)
points_matrix = np.dstack((X, Y))
Z = q(norm, points_matrix, b)
plt.imshow(Z, cmap='viridis', interpolation='nearest')
ax = plt.axes([0.3, 0.8, 0.3, 0.03])
Slider(ax=ax, valmin=-1000, valmax=1000, valinit=b, valstep=1, label='b');
ax = plt.axes([0.3, 0.7, 0.3, 0.03])
Slider(ax=ax, valmin=-1000, valmax=1000, valinit=p, valstep=1, label='p');
# Plot the surface

plt.show()
# %%
print(Z.shape)
# %%
b=90
print(b+1)
print(b/2+1)
# %%
