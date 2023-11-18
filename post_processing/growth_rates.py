# module to compute and plot time-average growth rates and real frequencies vs ky 
# can be imported into another script or run as a standalone script with
# > python growth_rates.py [list of .nc files]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys
from netCDF4 import Dataset

def growth_rates(fname, ikx=0, navgfac=0.5, label=None, plot=True, ax=None, Lref="a", refsp="i"):
    # read data from file
    data = Dataset(fname, mode='r')
    t = data.groups['Grids'].variables['time'][:]
    ky = data.groups['Grids'].variables['ky'][1:]
    omegas = data.groups['Diagnostics'].variables['omega_kxkyt'][:,1:,ikx,0]
    gams = data.groups['Diagnostics'].variables['omega_kxkyt'][:,1:,ikx,1]

    # compute time-average 
    istart_avg = int(len(t)*navgfac)
    omavg = np.mean(omegas[istart_avg:, :], axis=0)
    gamavg = np.mean(gams[istart_avg:, :], axis=0)
    if label == None:
        label = fname

    # plot growth rates and frequencies vs ky
    if plot:
        if ax.any() == None:
            fig, ax = plt.subplots(2)
        
        ax[0].plot(ky, gamavg, 'o', fillstyle='none')
        ax[1].plot(ky, omavg, 'o', fillstyle='none', label=label)
        
        ax[0].xaxis.set_minor_locator(AutoMinorLocator())
        ax[1].xaxis.set_minor_locator(AutoMinorLocator())
        
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[0].set_xlim(left=0)
        ax[1].set_xlim(left=0)
        bottom, top = ax[0].get_ylim()
        if bottom > 0:
            bottom = 0
            ax[0].set_ylim(bottom, top)
        bottom, top = ax[1].get_ylim()
        if bottom > 0:
            bottom = 0
            ax[1].set_ylim(bottom, top)
        ax[0].set_ylabel(r"$\gamma\ %s / v_{t%s}$"%(Lref, refsp))
        ax[1].set_ylabel(r"$\omega\ %s / v_{t%s}$"%(Lref, refsp))
        ax[1].set_xlabel(r"$k_y \rho_%s$"%refsp)
        legend = ax[1].legend()
        legend.set_in_layout(False)
        plt.tight_layout()

    return omavg, gamavg

if __name__ == "__main__":
    
    fig, ax = plt.subplots(nrows=2, num=0)

    for fname in sys.argv[1:]:
    
        try:
            growth_rates(fname, ax=ax)
        
        except:
            print(' usage: python growth_rates.py [list of .nc files]')
    
    #plt.savefig("growth_rates.png")
    plt.show()
