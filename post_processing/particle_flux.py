# module to compute and plot time-average particle flux
# can be imported into another script or run as a standalone script with
# > python particle_flux.py [list of .nc files]

import numpy as np
import matplotlib.pyplot as plt
import sys
from netCDF4 import Dataset

def particle_flux(data, ispec=0, navgfac=0.5, label=None, plot=True, fig=None, Lref="a", refsp=None):
    # read data from file
    t = data.groups['Grids'].variables['time'][:]
    try:
        p = data.groups['Diagnostics'].variables['ParticleFlux_st'][:,ispec]
    except:
        print('Error: particle flux data was not written. Make sure to use \'fluxes = true\' in the input file.')
    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"
    if refsp == None:
        refsp = species_tag

    # compute time-average and std dev
    istart_avg = int(len(t)*navgfac)
    pavg = np.mean(p[istart_avg:])
    pstd = np.std(p[istart_avg:])
    if label == None:
        label = data.filepath()
    print(r"%s: Gam_%s/Gam_GB = %.5g +/- %.5g" % (label, species_tag, pavg, pstd))

    # make a Gamma vs time plot
    if plot:
        if fig == None:
            fig = plt.figure(0)
        plt.plot(t,p,'-',label=r"%s: $\Gamma_%s/\Gamma_\mathrm{GB}$ = %.5g"%(label, species_tag, pavg))
        plt.ylabel(r"$\Gamma/\Gamma_\mathrm{GB}$")
        plt.xlabel(r"$t\ (v_{t%s}/%s)$"%(refsp, Lref))
        legend = plt.legend(loc='upper left')
        legend.set_in_layout(False)
        plt.tight_layout()

if __name__ == "__main__":
    
    print("Plotting particle fluxes.....")
    for fname in sys.argv[1:]:
        try:
            data = Dataset(fname, mode='r')
        except:
            print(' usage: python particle_flux.py [list of .nc files]')

        nspec = data.dimensions['s'].size

        if nspec <= 2:
            spec_list = [0]
        else:
            spec_list = np.arange(nspec)
    
        for ispec in spec_list:
            particle_flux(data, ispec=ispec, refsp="i")
    
    plt.xlim(0)
    # uncomment this line to save a PNG image of the plot
    plt.savefig("particle_flux.png", dpi=300)
    plt.show()
