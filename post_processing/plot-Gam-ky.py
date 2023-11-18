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
  if nspec == 2:
    nspec = 1 # ambipolar, so only plot one species
  label = data.filepath()
  for ispec in np.arange(nspec):
    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"
    Gamkyt = data.groups['Diagnostics'].variables['ParticleFlux_kyst'][:,ispec,:]
    Gamky = np.mean(Gamkyt[-int(len(t)/2):], axis=0)
    plt.plot(ky, Gamky, 'o-', label=r"%s, $\Gamma_%s$"%(label, species_tag))

refsp = 'i'
plt.xlabel(r'$k_y \rho_{%s}$' % refsp)
plt.ylabel(r"$\Gamma/\Gamma_\mathrm{GB}$")
#plt.xscale('log')
#plt.gca().set_yscale('symlog')
plt.legend()
plt.tight_layout()
plt.savefig("Gamky.png", dpi=300)
plt.show()
