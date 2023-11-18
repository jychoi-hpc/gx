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
  dky = ky[1] - ky[0]
  Akxt = data.groups['Diagnostics'].variables['Phi2_kxt'][:,:]
  Akx = np.mean(Akxt[-int(len(t)/2):], axis=(0))/dky
  maxP = max(maxP, np.max(Akx[1:]))
  plt.plot(kx, Akx, 'o-', label=fname)

plt.xscale('log')
plt.yscale('log')
plt.plot(kx, (kx/.3)**(-7/3)*maxP, 'k--', label=r"k^(-7/3)")
#plt.gca().set_ylim(top = 2*maxP)
plt.xlabel(r'$k_x \rho_i$')
plt.ylabel(r"$|\Phi|^2$")
plt.legend()
plt.tight_layout()
plt.savefig("Phi2kx.png", dpi=300)
plt.show()
