import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

i=0
plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  z = data.groups['Grids'].variables['theta'][:]
  dz = z[1] - z[0]
  try:
    scale = data.groups['Geometry'].variables['theta_scale'][:]
  except:
    try:
      stem = fname[:-7]
      geofile = glob.glob(f'{stem}_*wout*.nc')[0]
      geodata = Dataset(geofile, mode='r')
      scale = geodata.variables['scale'][:]
    except:
      print("no theta_scale data. assuming theta_scale = 1")
      scale = 1
  At = data.groups['Diagnostics'].variables['Phi2_zonal_zt'][:,:]
  A = np.mean(At[int(len(t)/2):,:], axis=(0))/dz
  plt.plot(z*scale, A, '-', label=fname)
  i+=1

plt.yscale('log')
#plt.ylim(top=np.max(Akx)*2)
plt.xlabel(r'$z/a$')
plt.ylabel(r"$|\Phi(ky=0)|^2$")
#plt.legend()
plt.tight_layout()
plt.show()
