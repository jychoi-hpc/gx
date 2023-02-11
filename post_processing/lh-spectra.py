import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset

def lh_spectra(data, ispec=0, navgfac=0.5, label=None, plot=True, fig=None):

    t = data.variables['time'][:]
    try:
        wlm = data.groups['Spectra'].variables['Wlmst'][:,ispec,:,:]
        nm = wlm.shape[1]
        nl = wlm.shape[2]
        
        wm = np.mean(wlm[int(len(t)/2):,...], axis=(0,2))*nl
        wl = np.mean(wlm[int(len(t)/2):,...], axis=(0,1))*nm
    except:
        try:
            wl = data.groups['Spectra'].variables['Wlst'][:,ispec,:,:]
            wm = data.groups['Spectra'].variables['Wmst'][:,ispec,:,:]
            nm = wm.shape[1]
            nl = wl.shape[1]
        except:
            print('Error: W(l,m) spectra data was not written.')

    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"

    if label == None:
        label = data.filepath()

    if plot:
        if fig == None:
            fig = plt.figure(ispec)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.plot(wm,'.-',label=label)
        ax1.set_xlabel("$m$")
        ax1.set_yscale('log')
        ax1.grid()
        ax1.set_title(r"$W_{g,%s}(m)$" % (species_tag))
        ax2.plot(wl,'.-',label=label)
        ax2.set_xlabel("$\ell$")
        ax2.set_yscale('log')
        ax2.grid()
        ax2.set_title(r"$W_{g,%s}(\ell)$" % (species_tag))
        ax2.legend()

if __name__ == "__main__":
    
    print("Plotting Hermite-Laguerre spectra.....")
    for fname in sys.argv[1:]:
        try:
            data = Dataset(fname, mode='r')
        except:
            print(' usage: python lh-spectra.py [list of .nc files]')

        nspec = data.dimensions['s'].size
    
        for ispec in np.arange(nspec):
            lh_spectra(data, ispec=ispec)
        
    # uncomment this line to save a PNG image of the plot
    #plt.savefig("lh_spectra.png")
    plt.show()

