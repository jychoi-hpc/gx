import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
import glob

from netCDF4 import Dataset

if sys.argv[-1].isnumeric():
  sidx = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  sidx = 0
  files = sys.argv[1:]

i=0
plt.figure(0)
for fname in files:
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
  Qt = data.groups['Diagnostics'].variables['HeatFlux_zst'][:,sidx,:]
  Q = np.mean(Qt[int(len(t)/2):,:], axis=(0))/dz
  plt.plot(z*scale, Q, '-', label=fname)

#  f = interp1d(z*scale, Q, fill_value='extrapolate')
#  if scale>2:
#      Qint = quad(f, -2*np.pi, 2*np.pi)[0]/2
#  else:
#      Qint = quad(f, -np.pi, np.pi)[0]
#  Qint = quad(f, -np.pi, np.pi)[0]
#  print(f"int(Q(z), -n*pi, n*pi) = {Qint}")
  i+=1

plt.yscale('log')
#plt.ylim(top=np.max(Akx)*2)
plt.xlabel(r'$\theta$')
plt.ylabel(r"$Q/Q_{GB}$")
plt.legend()
plt.tight_layout()
plt.savefig('Qz.png', dpi=300)
plt.show()
