import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
if sys.argv[-1].isnumeric():
  ikx = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  ikx = None
  files = sys.argv[1:]

for fname in files:
  data = Dataset("%s"%fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  kx = data.groups['Grids'].variables['kx'][:]
  if ikx is None:
    for i in np.arange(0, len(kx)):
      y = data.groups['Diagnostics'].variables['Phi2_kxt'][:,i]
      if i==int(len(kx)/2):
         fmt = '--'
      else:
         fmt = '-'
      plt.plot(t, y, fmt, label='kx = %.3f' % kx[i])
  else:
    y = data.groups['Diagnostics'].variables['Phi2_kxt'][:,ikx]
    plt.plot(t, y, '-', label='kx = %.3f' % kx[ikx])

plt.yscale('log')
plt.legend()

plt.show()
