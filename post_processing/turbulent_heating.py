# module to compute and plot time-average turbulent heating
# can be imported into another script or run as a standalone script with
# > python turbulent_heating.py [list of .nc files]

import numpy as np
import matplotlib.pyplot as plt
import sys
from netCDF4 import Dataset

def turbulent_heating(data, ispec=0, navgfac=0.5, label=None, plot=True, fig=None, Lref="a", refsp=None):
    # read data from file
    t = data.groups['Grids'].variables['time'][:-1]
    try:
        q = data.groups['Diagnostics'].variables['TurbulentHeating_st'][:-1,ispec]
    except:
        print('Error: turbulent heating data was not written. Make sure to use \'fluxes = true\' in the input file.')
    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"
    if refsp == None:
        refsp = species_tag

    # compute time-average and std dev
    istart_avg = int(len(t)*navgfac)
    qavg = np.mean(q[istart_avg:])
    qstd = np.std(q[istart_avg:])
    if label == None:
        label = data.filepath()
    print(r"%s: H_%s/H_GB = %.5g +/- %.5g" % (label, species_tag, qavg, qstd))

    # make a H vs time plot
    if plot:
        if fig == None:
            fig = plt.figure(0)
        plt.plot(t,q,'-',label=r"%s: $H_%s/H_\mathrm{GB}$ = %.5g"%(label, species_tag, qavg))
        plt.ylabel(r"$H/H_\mathrm{GB}$")
        plt.xlabel(r"$t\ (v_{t%s}/%s)$"%(refsp, Lref))
        legend = plt.legend(loc='upper left', ncols=1)
        legend.set_in_layout(False)
        plt.tight_layout()

if __name__ == "__main__":
    
    print("Plotting turbulent heating.....")
    for fname in sys.argv[1:]:
        try:
            data = Dataset(fname, mode='r')
        except:
            print(' usage: python turbulent_heating.py [list of .nc files]')

        nspec = data.dimensions['s'].size
    
        for ispec in np.arange(nspec):
            turbulent_heating(data, ispec=ispec, refsp="i")
    
    plt.xlim(0)
    # uncomment this line to save a PNG image of the plot
    plt.tight_layout()
    plt.savefig("turbulent_heating.png", dpi=300)
    plt.show()
