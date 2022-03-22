import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

for fname in sys.argv[1:]:
  data = Dataset("%s.nc"%fname, mode='r')
  t = data.variables['time'][:]
  ky = data.variables['ky'][:]
  for i in np.arange(1, len(ky)):
    y = data.groups['Spectra'].variables['Pkyst'][:,0,i]
    plt.plot(t, y, label=fname)
    
    #fit = stats.linregress(t, np.log(y))
    #plt.plot(t, np.exp(fit.intercept + fit.slope*t), '--', label=r'%s, ky = %.3f: $\gamma = $%.5f' % (fname, ky[i], fit.slope/2))
    #print("%s, ky = %f: gamma = %f" %( fname, ky[i], fit.slope/2))

plt.yscale('log')
plt.legend()
plt.show()
