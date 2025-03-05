import os
import sys
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt

fname = sys.argv[1]
field = sys.argv[2]
data = Dataset(fname, 'r')

x = data.groups['Grids'].variables['x'][:]
y = data.groups['Grids'].variables['y'][:]
z = data.groups['Grids'].variables['theta'][:]
print(x)
print(y)
nz = z.shape[0]

time = data.groups['Grids'].variables['time'][:]
fieldXY = data.groups['Diagnostics'].variables[f'{field}XY'][:].data[:,:,:,nz//2]

def animate(self, i):
    im = plt.pcolormesh(x, y, fieldXY[i], cmap='inferno')
    plt.title(f'$t={time[i]} (v_t/a)$')

fig = plt.figure()
plt.hold(True)
plt.pcolormesh(x, y, fieldsXY[0], cmap='inferno')

anim = animation.FuncAnimation(fig, animate, frames = range(2,10), blit = False)

plt.show()
plt.hold(False)
