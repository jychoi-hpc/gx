import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
maxP = 0
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  kx = data.groups['Grids'].variables['kx'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  dkx = kx[1] - kx[0]
  Pkyt = data.groups['Diagnostics'].variables['Phi2_kyt'][:,:]
  Pky = np.mean(Pkyt[-int(len(t)/2):], axis=0)/dkx
  maxP = max(maxP, np.max(Pky[1:]))
  plt.plot(ky, Pky, 'o-', label=fname)

plt.xscale('log')
plt.yscale('log')
plt.plot(ky, (ky/.3)**(-7/3)*maxP, 'k--', label=r"k^(-7/3)")
plt.gca().set_ylim(top = 2*maxP)
plt.xlabel(r'$k_y \rho_i$')
plt.ylabel(r"$|\Phi|^2$")
plt.legend()
plt.tight_layout()
plt.savefig("Phi2ky.png", dpi=300)
plt.show()
