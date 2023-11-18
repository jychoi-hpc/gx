import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
if sys.argv[-1].isnumeric():
  iky = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  iky = None
  files = sys.argv[1:]

for fname in files:
  data = Dataset("%s"%fname, mode='r')
  t = data.groups['Grids'].variables['time'][:]
  ky = data.groups['Grids'].variables['ky'][:]
  if iky is None:
    for i in np.arange(0, len(ky)):
      y = data.groups['Diagnostics'].variables['Phi2_kyt'][:,i]
      if i==0:
         fmt = '--'
      else:
         fmt = '-'
      plt.plot(t, y, fmt, label='ky = %.3f' % ky[i])
  else:
    y = data.groups['Diagnostics'].variables['Phi2_kyt'][:,iky]
    plt.plot(t, y, '-', label='ky = %.3f' % ky[iky])

    #fit = stats.linregress(t[int(len(t)/2):-1], np.log(y[int(len(t)/2):-1]))
    ##plt.plot(t, np.exp(fit.intercept + fit.slope*t), '--', label=r'%s, ky = %.3f: $\gamma = $%.5f' % (fname, ky[i], fit.slope/2))
    #print("%s, ky = %f: gamma = %f" %( fname, ky[i], fit.slope/2))

plt.yscale('log')
plt.legend()

#plt.figure(1)
#avg = np.mean(data.groups['Spectra'].variables['Pkyst'][:,0,:], axis=0)
#plt.plot(ky, avg, 'o-')
#plt.xscale('log')
#plt.yscale('log')
plt.show()
