# Script to check and plot |Phi|**2 vs time for kinetic alfven wave against "correct" restults
# Usage:
# > python check.py [input file stem]
# Example:
# > python check.py kaw_betahat10.0_kp0.01
# This will check kaw_betahat10.0_kp0.01.nc against kaw_betahat10.0_kp0.01_correct.nc
# The script will print to screen the integrated relative difference in |Phi|**2,
# and also generate a plot comparing the results.

import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats as stats
import sys

from netCDF4 import Dataset

stem = sys.argv[1]

data = Dataset("%s.out.nc" % stem, mode='r')
t = data.groups['Grids'].variables['time'][:]
ky = data.groups['Grids'].variables['ky'][:]
Pky = data.groups['Diagnostics'].variables['Phi2_kyt'][:,1]
plt.plot(t, Pky, label='GX')

data = Dataset("%s_correct.out.nc" % stem, mode='r')
t = data.groups['Grids'].variables['time'][:]
ky = data.groups['Grids'].variables['ky'][:]
check_Pky = data.groups['Diagnostics'].variables['Phi2_kyt'][:,1]
plt.plot(t, check_Pky, label='correct')

dt = t[1]-t[0]
diff = np.abs(Pky - check_Pky)/check_Pky
diffsum = sum(diff*dt)
if ( diffsum < 5.e-3 ):
    checkstr = "TEST PASSES\n"
else:
    checkstr = "TEST FAILS\n"

print("\nIntegrated relative difference in |Phi|**2 vs time = %e... %s" % (diffsum, checkstr))

#fit = stats.linregress(t, np.log(y))
#plt.plot(t, np.exp(fit.intercept + fit.slope*t), '--', label=r'%s, ky = %.3f: $\gamma = $%.5f' % (fname, ky[i], fit.slope/2))
#print("%s, ky = %f: gamma = %f" %( fname, ky[i], fit.slope/2))

plt.yscale('log')
plt.legend()
plt.show()
