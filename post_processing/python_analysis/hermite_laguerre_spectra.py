import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
import sys
from load_files import load_files
from extract_species import extract_species


def plot_hermite_laguerre_spectra(simulations, average_fraction=0.5, time=False):

    plt.close("all")

    print("")
    print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []
    
    # Initialising plotting
    fig, ((ax1), (ax2)) = plt.subplots(1, 2)
    fig.canvas.manager.set_window_title("Hermite-laguerre spectra (all simulations)")
    plot_color = plt.cm.rainbow(np.linspace(0, 1, len(simulations.keys())))

    # Iterating over similation files 
    for simulation_index, simulation_key in enumerate(simulations.keys()):

        simulation = simulations[simulation_key]

        t_range  = simulation['Dimensions']['time']
        it_start = int(len(t_range) * average_fraction)

        try:
            hermite_spectrum  = np.sum(simulation['Spectra']['Wlmst'], axis=3)
            laguerre_spectrum = np.sum(simulation['Spectra']['Wlmst'], axis=2)

        except:
            try:
                hermite_spectrum  = simulation['Spectra']['Wmst']
                laguerre_spectrum = simulation['Spectra']['Wlst']

            except:
                print("%s -- WARNING: Hermite-laguerre spectra not found." %(simulation_key.name))
                continue

        hermite_spectrum_avg  = np.average(hermite_spectrum[it_start:, :, :], axis=0)
        laguerre_spectrum_avg = np.average(laguerre_spectrum[it_start:, :, :], axis=0)

        # Check whether simulation has same reference species
        species_list, ref_species = extract_species(simulation)
        ref_species_list.append(ref_species)

        if not (ref_species == ref_species_list[0]):
            print("%s -- WARNING: Inconsistent reference species." %(simulation_key.name))
            continue

        if ref_species == "ions":
            linestyles_dict = {"ions": "solid", "electrons": "dashed"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))
        else:
            linestyles_dict = {"ions": "dashed", "electrons": "solid"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))

        # Plotting 
        for species_index, species in enumerate(species_list):

            print(r"%s -- maxima at m = %d, l = %d" % (simulation_key.name, np.argmax(hermite_spectrum_avg[species_index]), np.argmax(laguerre_spectrum_avg[species_index])))
        
            if species_index == 0:
                plot_label = simulation_key.name
            else:
                plot_label = None

            ax1.plot(hermite_spectrum_avg[species_index, :], label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])
            ax2.plot(laguerre_spectrum_avg[species_index, :], label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])

        # Plotting Hermite-Laguerre spectra as a function of time 
        if time:
            fig, ((ax3), (ax4)) = plt.subplots(1, 2)
            fig.canvas.manager.set_window_title("Hermite-laguerre spectra (%s)" % simulation_key.name)

            normalize_time = matplotlib.colors.Normalize(vmin=0, vmax=t_range.max())
            scalarmappaple = matplotlib.cm.ScalarMappable(norm = normalize_time, cmap = "rainbow")
            plot_colour_it = plt.cm.rainbow(np.linspace(0, 1, len(t_range)))
            colorbar = plt.colorbar(scalarmappaple, ax=ax4)
            colorbar.set_label(r"$t(\sqrt{2}a/v_{\mathrm{th}%s})$" % ref_species[0], labelpad = 15)
            
            for species_index, species in enumerate(species_list):
                for it in range(len(t_range)):
                    ax3.plot(hermite_spectrum[it, species_index, :], linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_colour_it[it])
                    ax4.plot(laguerre_spectrum[it, species_index, :], linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_colour_it[it])

            ax3.set_xscale("linear")
            ax3.set_xlabel(r"Hermite moment, $m$")
            ax3.set_yscale("log")
            ax3.set_ylabel(r"$W_{s}(m)$")

            ax4.set_xscale("linear")
            ax4.set_xlabel(r"Laguerre moment, $\ell$")
            ax4.set_yscale("log")
            ax4.set_ylabel(r"$W_{s}(\ell)$")

    # Setting plot options
    ax1.set_xscale("linear")
    ax1.set_xlabel(r"Hermite moment, $m$")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$W_{s}(m)$")

    ax2.set_xscale("linear")
    ax2.set_xlabel(r"Laguerre moment, $\ell$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$W_{s}(\ell)$")

    ax2.legend(handlelength = 0, title = "Simulations")

    print("")
    print("Linestyles:", linestyles_dict)
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

    time = False

    if "time" in filenames:
        filenames.remove("time")
        time = True

    simulations = load_files(filenames, groups = ['Inputs', 'Spectra'], spectra = ['Wmst', 'Wlst', 'Wlmst'])
    
    plot_hermite_laguerre_spectra(simulations, time=time)