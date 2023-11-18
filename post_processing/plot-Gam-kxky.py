import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
import sys

from netCDF4 import Dataset

for fname in sys.argv[1:]:
  fig = plt.figure()
  plt.inferno()
  log = True
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  kx = data.groups['Grids'].variables['kx'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  Gamkxkyt = data.groups['Diagnostics'].variables['ParticleFlux_kxkyst'][:,0,:,:]
  Gamkxky = np.mean(Gamkxkyt[int(len(t)/2):], axis=0)
  if log:
     lognorm = colors.LogNorm(vmin=1e-4, vmax=Gamkxky.max())
  else:
     lognorm = None
  pc = plt.pcolormesh(ky, kx, Gamkxky.T, norm=lognorm)
  fig.colorbar(pc)

  refsp = 'i'
  plt.xlabel(r'$k_y \rho_{%s}$' % refsp)
  plt.ylabel(r'$k_x \rho_{%s}$' % refsp)
  plt.title(r"$\Gamma/\Gamma_\mathrm{GB}$")
  plt.gca().set_aspect('equal')
  plt.tight_layout()
  plt.savefig(f"{fname[:-7]}_Gamkxky.png", dpi=300)
plt.show()
