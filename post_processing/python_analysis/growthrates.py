import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from itertools import cycle, islice
import sys
from load_files import load_files
from extract_species import extract_species


def plot_growthrates(simulations, average_fraction=0.5, kx_default=0.0, kx_plots=False):

    plt.close("all")

    print("")
    print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []
    
    # Initialising plotting
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.manager.set_window_title("Growthrates and frequencies (all simulations)")
    plot_color = plt.cm.rainbow(np.linspace(0, 1, len(simulations.keys())))

    # Iterating over similation files 
    for simulation_index, simulation_key in enumerate(simulations.keys()):

        simulation = simulations[simulation_key]

        t_range     = simulation['Dimensions']['time']
        ky_range    = simulation['Dimensions']['ky'][:]
        kx_range    = simulation['Dimensions']['kx'][:]
        ikx_default = len(kx_range)//2 + int(np.nan_to_num((kx_default/kx_range[-1]))*len(kx_range))

        # Extracting/interpolating growthrates
        try:
            omegas     = simulation['Special']['omega_v_time'][:, :, :, 0]
            gammas     = simulation['Special']['omega_v_time'][:, :, :, 1]

            it_start_average = int(len(t_range) * average_fraction)
            omegas_avg = np.average(omegas[it_start_average:, :, :], axis = 0)
            gammas_avg = np.average(gammas[it_start_average:, :, :], axis = 0)
            
        except:
            print("%s -- WARNING: 'omega_v_time' not found not found in output file. Attempting to interpolate growthrates." %(simulation_key.name))
            nt_interp  = 5
            amplitudes = np.zeros(nt_interp)
            gammas_num = np.zeros([len(ky_range), len(kx_range)])

            try:
                Phi2kxkyt = simulation['Spectra']['Phi2kxkyt']
            except:
                print("%s -- WARNING: Spectra required for interpolation (Phi2kxkyt) not found." %(simulation_key.name))
                continue

            for ikx in range(len(kx_range)):
                for iky in range(len(ky_range)):
                    for it in range(-1 - nt_interp, -1):
                        amplitudes[it + 1] = Phi2kxkyt[it, iky, ikx]
                    slope , constant = np.polyfit(t_range[-1 - nt_interp: -1], np.log(amplitudes), 1) 
                    if np.isnan(slope):
                        gammas_num[iky, ikx] = 0
                    else:
                        gammas_num[iky, ikx] = slope/2

            omegas_avg = np.zeros([len(ky_range), len(kx_range)])  
            gammas_avg = gammas_num

        # Check whether simulation has same reference species
        species_list, ref_species = extract_species(simulation)
        ref_species_list.append(ref_species)

        if not (ref_species == ref_species_list[0]):
            print("%s -- WARNING: Inconsistent reference species." %(simulation_key.name))
            continue

        # Plotting
        print(r"%s -- maximum growthrate %.4g at ky = %.4g" % (simulation_key.name, np.max(gammas_avg[:, ikx_default]), ky_range[np.argmax(gammas_avg[:, ikx_default])]))

        plot_label = simulation_key.name
        
        ax1.plot(ky_range, gammas_avg[:, ikx_default], label=plot_label, linewidth=2, marker='o', color=plot_color[simulation_index])
        ax2.plot(ky_range, omegas_avg[:, ikx_default], label=plot_label, linewidth=2, marker='o', color=plot_color[simulation_index])

        # Plotting growthrates as a function of kx
        if kx_plots:
            fig, (ax3, ax4) = plt.subplots(2, 1)
            fig.canvas.manager.set_window_title("Growthrates and frequencies (%s)" % simulation_key.name)

            normalize_kx   = matplotlib.colors.Normalize(vmin=-kx_range.max(), vmax=kx_range.max())
            scalarmappaple = matplotlib.cm.ScalarMappable(norm=normalize_kx, cmap="rainbow")
            plot_colour_kx = plt.cm.rainbow(np.linspace(0, 1, len(kx_range)))
            colorbar_ax3   = plt.colorbar(scalarmappaple, ax=ax3)
            colorbar_ax4   = plt.colorbar(scalarmappaple, ax=ax4)
            colorbar_ax3.set_label(r"$k_y \rho_{%s}$" %(ref_species[0]), labelpad = 15)
            colorbar_ax4.set_label(r"$k_y \rho_{%s}$" %(ref_species[0]), labelpad = 15)
            
            for ikx in range(len(kx_range)):
                ax3.plot(ky_range, gammas_avg[:, ikx], linewidth=2, marker='o', color=plot_colour_kx[ikx])
                ax4.plot(ky_range, omegas_avg[:, ikx], linewidth=2, marker='o', color=plot_colour_kx[ikx])

            ax3.set_xscale("log")
            ax3.set_xlabel(None)
            ax3.set_yscale("linear")
            ax3.set_ylabel(r"$\gamma /(v_{th%s}/a)$" %(ref_species[0]))

            ax4.set_xscale("log")
            ax4.set_xlabel(r"$k_y \rho_{%s}$" %(ref_species[0]))
            ax4.set_yscale("linear")
            ax4.set_ylabel(r"$\omega /(v_{th%s}/a)$" %(ref_species[0]))

    
    # Setting plot options
    ax1.set_xscale("log")
    ax1.set_xlabel(None)
    ax1.set_yscale("linear")
    ax1.set_ylabel(r"$\gamma /(v_{th%s}/a)$" %(ref_species[0]))

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k_y \rho_{%s}$" %(ref_species[0]))
    ax2.set_yscale("linear")
    ax2.set_ylabel(r"$\omega /(v_{th%s}/a)$" %(ref_species[0]))

    ax1.legend(handlelength = 0, title = "Simulations")

    print('─' * 100)
    print("Completed.")
    print("")

    plt.show()
    
    return

if __name__ == "__main__":

    # Interpreting command line input
    filenames = sys.argv[1:]

    if "latex" in filenames:
        filenames.remove("latex")

        font = {'family' : 'serif',
        'serif'  : ['Computer Modern Roman'],
        'weight' : 'bold',
        'size'   : 18} # Use 40 with latex for papers, 18 otherwise

        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['axes.labelpad']='20'

        print("")
        print("Using LaTeX")

    kx_plots = False

    if "kx_plots" in filenames:
        filenames.remove("kx_plots")
        kx_plots = True

    simulations = load_files(filenames, groups = ['Inputs', 'Special', 'Spectra'], spectra = ['Phi2kxkyt'])
    
    plot_growthrates(simulations, kx_plots=kx_plots)

