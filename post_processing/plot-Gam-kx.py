import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.variables['time'][:]
  kx = data.variables['kx'][:]
  Gamkxt = data.groups['Spectra'].variables['Gamkxst'][:,0,:]
  Gamkx = np.mean(Gamkxt[-int(len(t)/2):], axis=0)
  plt.plot(kx, Gamkx, 'o-')

refsp = 'i'
plt.xlabel(r'$k_x \rho_{%s}$' % refsp)
plt.ylabel(r"$\Gamma/\Gamma_\mathrm{GB}$")
#plt.xscale('log')
#plt.gca().set_yscale('symlog')
plt.tight_layout()
plt.show()

