#!/usr/bin/env python3
import numpy as np
from netCDF4 import Dataset as ds
from matplotlib import pyplot as plt

import pdb
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

rtg1 = ds(f1).groups["Geometry"]
scale = rtg1.variables["theta_scale"][:].data
tgrid_1 = ds(f1).groups["Grids"].variables["theta"][:].data*scale/np.pi
bmag_1 = rtg1.variables["bmag"][:].data
gds2_1 = rtg1.variables["gds2"][:].data
gds22_1 = rtg1.variables["gds22"][:].data
gds21_1 = rtg1.variables["gds21"][:].data
cvdrift_1 = rtg1.variables["cvdrift"][:].data
gbdrift_1 = rtg1.variables["gbdrift"][:].data
cvdrift0_1 = rtg1.variables["cvdrift0"][:].data
grho_1 = rtg1.variables["grho"][:].data
gradpar_1 = rtg1.variables["gradpar"][:].data
drhodpsi_1 = rtg1.variables["drhodpsi"][:].data

rtg2 = ds(f2).groups["Geometry"]
scale = rtg2.variables["theta_scale"][:].data
tgrid_2 = ds(f2).groups["Grids"].variables["theta"][:].data*scale/np.pi
gds2_2 = rtg2.variables["gds2"][:].data
bmag_2 = rtg2.variables["bmag"][:].data
gds22_2 = rtg2.variables["gds22"][:].data
gds21_2 = rtg2.variables["gds21"][:].data
cvdrift_2 = rtg2.variables["cvdrift"][:].data
gbdrift_2 = rtg2.variables["gbdrift"][:].data
cvdrift0_2 = rtg2.variables["cvdrift0"][:].data
grho_2 = rtg2.variables["grho"][:].data
gradpar_2 = rtg2.variables["gradpar"][:].data
drhodpsi_2 = rtg2.variables["drhodpsi"][:].data

print(gradpar_1, gradpar_2)


fig, ((plt1, plt2, plt3, plt4), (plt5, plt6, plt7, plt8)) = plt.subplots(2,4, figsize=(13, 6))

plt1.plot(tgrid_1, cvdrift_1,'o-r', tgrid_2, cvdrift_2, 'o-g')
plt1.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt1.set_ylabel('cvdrift', fontsize=16)
#plt1.text(0.1, 0.2, 'scale=%.3f'%(np.max(gbdrift)/np.max(gbdrift2)))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt2.plot(tgrid_1, bmag_1,'o-r', tgrid_2, bmag_2, 'o-g')
plt2.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt2.set_ylabel('bmag', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt3.plot(tgrid_1, gds2_1, 'o-r', tgrid_2, gds2_2, 'o-g')
plt3.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt3.set_ylabel('gds2', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt5.plot(tgrid_1, gds22_1, 'o-r', tgrid_2, gds22_2, 'o-g')
plt5.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt5.set_ylabel('gds22', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt7.plot(tgrid_1, gbdrift_1, 'o-r', tgrid_2, gbdrift_2, 'o-g')
plt7.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt7.set_ylabel('gbdrift', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
#plt7.set_ylim([gradpar[0]-0.1, gradpar[0]+0.1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt4.plot(tgrid_1, gds21_1,'o-r', tgrid_2, gds21_2, 'o-g')
plt4.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt4.set_ylabel('gds21', fontsize=16)
#plt4.text(0.9, 0.1, 'scale=%.3f'%(np.max(gds21)/np.max(gds21_2)))
#plt4.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt6.plot(tgrid_1, grho_1, 'o-r', tgrid_2, grho_2, 'o-g')
plt6.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt6.set_ylabel('grho', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt8.plot(tgrid_1, cvdrift0_1, 'o-r', tgrid_2, cvdrift0_2, 'o-g')
plt8.set_xlabel(r'$\theta_{\mathrm{PEST}}/\pi$', fontsize=16)
plt8.set_ylabel('cvdrift0', fontsize=16)
#plt2.text(0.9, 0.2, 'scale=%.3f'%(np.max(bmag)/np.max(bmag2)))
#plt2.set_title('rho=0.25', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()


