import pdb
import os
import sys
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt

for i, fname in enumerate(sys.argv[1:-1]):
    iky = int(sys.argv[-1])

    rtg = Dataset(fname, 'r')
    tgrid = rtg.groups['Grids'].variables['theta'][:].data
    nz = tgrid.shape[0]
    tgrid = np.append(tgrid, -tgrid[0]) # periodic point
    ky = rtg.groups['Grids'].variables['ky'][:].data

    if i==0:
        nFields = 1
    try:
        phi  = np.asarray(rtg.groups['Diagnostics'].variables['Phi'][:].data[-1, iky,0,:,:])
        phi = np.vstack((phi, phi[0,:])) # periodic point
        phiC = phi[:,0] + 1j*phi[:,1]
        norm = phiC[int(nz/2)]
        phi  = np.transpose(np.array([(phiC /norm).real, (phiC /norm).imag]))
        phi_title = r'$q_\mathrm{ref}\Phi/T_\mathrm{ref}\ a_\mathrm{N}/\rho_\mathrm{ref}$'
    except:
        print("No fields have been written! Set fields=true in the [Diagnostics] section of the input file")
        exit(1)
    try:
        apar = rtg.groups['Diagnostics'].variables['Apar'][:].data[-1, iky,0,:,:]
        apar = np.vstack((apar, apar[0,:])) # periodic point
        aparC = apar[:,0] + 1j*apar[:,1]
        apar = np.transpose(np.array([(aparC/norm).real, (aparC/norm).imag]))
        hasApar = True
        if i==0:
            nFields += 1
        apar_title = r'$A_\parallel\ a_\mathrm{N}/(\rho_\mathrm{ref}^2 B_N)$'
    except:
        hasApar = False
    try:
        bpar = rtg.groups['Diagnostics'].variables['Bpar'][:].data[-1, iky,0,:,:]
        bpar = np.vstack((bpar, bpar[0,:])) # periodic point
        bparC = bpar[:,0] + 1j*bpar[:,1]
        bpar = np.transpose(np.array([(bparC/norm).real, (bparC/norm).imag]))
        hasBpar = True
        if i==0:
            nFields += 1
        bpar_title = r'$\delta B_\parallel/B\ a_\mathrm{N}/\rho_\mathrm{ref}$'
    except:
        hasBpar = False

    if nFields==1:
        if i==0:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax.plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax.set_title(phi_title)
        ax.set_xlabel(r'$\theta$')
    if nFields==2:
        if i==0:
            fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax[0].plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax[0].set_title(phi_title)
        if hasApar:
            ax[1].plot(tgrid, apar[:,0], '-', color=f'C{i}')
            ax[1].plot(tgrid, apar[:,1], '--', color=f'C{i}')
            ax[1].set_title(apar_title)
        if hasBpar:
            ax[1].plot(tgrid, bpar[:,0], '-', color=f'C{i}')
            ax[1].plot(tgrid, bpar[:,1], '--', color=f'C{i}')
            ax[1].set_title(bpar_title)
        ax[0].set_xlabel(r'$\theta$')
        ax[1].set_xlabel(r'$\theta$')
    if nFields==3:
        if i==0:
            fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0,0].plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax[0,0].plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax[0,0].set_title(phi_title)
        if hasApar:
            ax[0,1].plot(tgrid, apar[:,0], '-', color=f'C{i}')
            ax[0,1].plot(tgrid, apar[:,1], '--', color=f'C{i}')
            ax[0,1].set_title(apar_title)
        if hasBpar:
            ax[1,0].plot(tgrid, bpar[:,0], '-', color=f'C{i}')
            ax[1,0].plot(tgrid, bpar[:,1], '--', color=f'C{i}')
            ax[1,0].set_title(bpar_title)
        ax[1,1].axison = False

        ax[0,0].set_xlabel(r'$\theta$')
        ax[1,0].set_xlabel(r'$\theta$')
        ax[0,1].set_xlabel(r'$\theta$')

plt.tight_layout()
plt.show()

