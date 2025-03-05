import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
if sys.argv[-1].isnumeric():
  il = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  il = None
  files = sys.argv[1:]
isp = 1

for fname in files:
  data = Dataset("%s"%fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  nl = data.dimensions['l'].size
  if il is None:
    for i in np.arange(0, nl):
      y = np.mean(data.groups['Diagnostics'].variables['Wg_lmst'][:,isp,:,i], axis=1)
      plt.plot(t, y, '-', label='l = %d' % i)
  else:
    y = np.mean(data.groups['Diagnostics'].variables['Wg_lmst'][:,isp,:,il], axis=1)
    plt.plot(t, y, '-', label='l = %d' % il)

plt.yscale('log')
plt.legend()

plt.show()
