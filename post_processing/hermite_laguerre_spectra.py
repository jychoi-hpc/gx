import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset

def hermite_laguerre_spectra(data, ispec=0, navgfac=0.5, label=None, plot=True, axs=None, normalize=False):

    t = data.groups['Grids'].variables['time'][:]
    try:
        wlm = data.groups['Diagnostics'].variables['Wg_lmst'][:,ispec,:,:]
        nm = wlm.shape[1]
        nl = wlm.shape[2]
        
        wm = np.mean(wlm[int(len(t)/2):,...], axis=(0,2))*nl
        wl = np.mean(wlm[int(len(t)/2):,...], axis=(0,1))*nm
    except:
        print('Error: W(l,m) spectra data was not written. Use free_energy = true in [Diagnostics].')

    if normalize:
        wm = wm / wm[0]
        wl = wl / wl[0]

    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"

    if label == None:
        label = data.filepath()

    if plot:
        axs[0].plot(wm,'.-',label=label)
        axs[0].set_xlabel("Hermite mode, $m$")
        axs[0].set_yscale('log')
        axs[0].grid()
        axs[0].set_title(r"$W_{g,%s}(m)$" % (species_tag))
        axs[1].plot(wl,'.-',label=label)
        axs[1].set_xlabel("Laguerre mode, $\ell$")
        axs[1].set_yscale('log')
        axs[1].grid()
        axs[1].set_title(r"$W_{g,%s}(\ell)$" % (species_tag))
        axs[1].legend()

if __name__ == "__main__":
    
    print("Plotting Hermite-Laguerre spectra.....")
    fig = plt.figure(0, figsize=(10,10))
    axs = None

    for fname in sys.argv[1:]:
        try:
            data = Dataset(fname, mode='r')
        except:
            print(' usage: python hermite_laguerre_spectra.py [list of .nc files]')

        nspec = data.dimensions['s'].size
        if axs is None:
            axs = fig.subplots(nspec, 2)

        if nspec == 1:
            axs = np.reshape(axs,(1,-1))
    
        for ispec in np.arange(nspec):
            hermite_laguerre_spectra(data, ispec=ispec, axs=axs[ispec,:])
        
    # uncomment this line to save a PNG image of the plot
    #plt.savefig("hermite_laguerre_spectra.png")
    plt.tight_layout()
    plt.show()

