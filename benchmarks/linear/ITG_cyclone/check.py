# Script to check and plot growth rates against a "correct" set
# Usage:
# > python check.py [input file stem]
# Example:
# > python check.py itg_miller_adiabatic_electrons
# This will check itg_miller_adiabatic_electrons.nc against itg_miller_adiabatic_electrons_correct.nc
# The script will print to screen the maximum relative differences in growth rates and real frequencies,
# and also generate a plot comparing the results.

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.style as style
from matplotlib.ticker import AutoMinorLocator

import os

from netCDF4 import Dataset

fig, (ax1, ax2) = plt.subplots(2)
stem = sys.argv[1]
kmax = None

# read gx data
data = Dataset("%s.nc" % stem, mode='r')
t = data.variables['time'][:]
ky = data.variables['ky'][1:]
omegas = data.groups['Special'].variables['omega_v_time'][:,1:,0,0]
gams = data.groups['Special'].variables['omega_v_time'][:,1:,0,1]
omavg = np.mean(omegas[int(len(t)/2):, :], axis=0)
gamavg = np.mean(gams[int(len(t)/2):, :], axis=0)
ax1.plot(ky[:kmax], gamavg[:kmax], 'o', fillstyle='none')
ax2.plot(ky[:kmax], omavg[:kmax], 'o', fillstyle='none', label='GX')

check = Dataset("%s_correct.nc" % stem, mode='r')
t = check.variables['time'][:]
ky = check.variables['ky'][1:]
check_omegas = check.groups['Special'].variables['omega_v_time'][:,1:,0,0]
check_gams = check.groups['Special'].variables['omega_v_time'][:,1:,0,1]
check_omavg = np.mean(check_omegas[int(len(t)/2):, :], axis=0)
check_gamavg = np.mean(check_gams[int(len(t)/2):, :], axis=0)
ax1.plot(ky[:kmax], check_gamavg[:kmax], 's', fillstyle='none')
ax2.plot(ky[:kmax], check_omavg[:kmax], 's', fillstyle='none', label='correct')

diff_gams = (gamavg - check_gamavg)/check_gamavg
diff_omegas = (omavg - check_omavg)/check_omavg

max_diff_gams = np.max(np.abs(diff_gams))
max_diff_omegas = np.max(np.abs(diff_omegas))
if ( max_diff_gams < 1e-3 and max_diff_omegas < 5e-3 ):
    checkstr = "TEST PASSES\n"
else:
    checkstr = "TEST FAILS\n"

print("\nMaximum relative difference in growth rates = %e" % max_diff_gams)
print("Maximum relative difference in real frequencies = %e" % max_diff_omegas)
print(checkstr)

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())

ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax1.set_xlim(left=0)
ax2.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.set_ylabel(r'$\gamma\ a / v_{ti}$')
#ax1.set_xlabel(r'$k_y \rho_i$')
ax2.set_ylabel(r'$\omega\ a / v_{ti}$')
ax2.set_xlabel(r'$k_y \rho_i$')
ax2.legend()
plt.tight_layout()
plt.show()
