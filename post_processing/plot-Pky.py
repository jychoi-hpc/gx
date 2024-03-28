import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.variables['time'][:]
  ky = data.variables['ky'][:]
  Pkyt = data.groups['Spectra'].variables['Pkyst'][:,0,:]
  Pky = np.mean(Pkyt[int(len(t)/2):], axis=0)
  plt.plot(ky, Pky, 'o-')

plt.xscale('log')
plt.yscale('log')
plt.show()
