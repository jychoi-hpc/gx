import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  kx = data.groups['Grids'].variables['kx'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  dkx = kx[1] - kx[0]
  Akxky0t = data.groups['Diagnostics'].variables['Phi2_zonal_kxt'][:,:]
  Akx = np.mean(Akxky0t[-int(len(t)/2):], axis=0)/dkx
  plt.plot(kx, Akx, 'o-')

plt.xscale('log')
plt.yscale('log')
plt.ylim(top=np.max(Akx)*2)
plt.xlabel(r'$k_x \rho_i$')
plt.ylabel(r"$|\Phi|^2(k_y=0)$")
plt.legend()
plt.show()
