import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
if sys.argv[-1].isnumeric():
  im = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  im = None
  files = sys.argv[1:]
isp = 1

for fname in files:
  data = Dataset("%s"%fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  nm = data.dimensions['m'].size
  if im is None:
    for i in np.arange(0, nm):
      y = np.mean(data.groups['Diagnostics'].variables['Wg_lmst'][:,isp,i,:], axis=1)
      plt.plot(t, y, '-', label='m = %d' % i)
  else:
    y = np.mean(data.groups['Diagnostics'].variables['Wg_lmst'][:,isp,im,:], axis=1)
    plt.plot(t, y, '-', label='m = %d' % im)

plt.yscale('log')
plt.legend()

plt.show()
