import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
import sys

from netCDF4 import Dataset

fig = plt.figure(0)
plt.inferno()
log = True
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  kx = data.groups['Grids'].variables['kx'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  Pkxkyt = data.groups['Diagnostics'].variables['Phi2_kxkyt'][:,:,:]
  Pkxky = np.mean(Pkxkyt[int(len(t)/2):], axis=0)
  if log:
     lognorm = colors.LogNorm(vmin=1e-4, vmax=1e-1)
  else:
     lognorm = None
  pc = plt.pcolormesh(ky, kx, Pkxky.T, norm=lognorm)
  fig.colorbar(pc)

refsp = 'i'
plt.xlabel(r'$k_y \rho_{%s}$' % refsp)
plt.ylabel(r'$k_x \rho_{%s}$' % refsp)
plt.title(r"$|\Phi|^2$")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
