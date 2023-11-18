import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  nspec = data.dimensions['s'].size
  label = data.filepath()
  for ispec in np.arange(nspec):
    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"
    Qkyt = data.groups['Diagnostics'].variables['HeatFlux_kyst'][:,ispec,:]
    Qky = np.mean(Qkyt[int(len(t)/2):], axis=0)
    plt.plot(ky, Qky, 'o-', label=r"%s, $Q_%s$"%(label, species_tag))

refsp = 'i'
plt.xlabel(r'$k_y \rho_{%s}$' % refsp)
plt.ylabel(r"$Q/Q_\mathrm{GB}$")
#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("Qky.png", dpi=300)
plt.show()
