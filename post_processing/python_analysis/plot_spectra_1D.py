import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
from extract_species import extract_species

def plot_spectra_1D(simulations, spectrum, spectrum_label, spectrum_range, spectrum_range_label, spectrum_xscale, spectrum_yscale, average_fraction=0.5, absolute_value=False, time_plots=False):

    plt.close("all")

    # print("")
    print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []
    electromagnetic_spectra_list = ['Phi2kyt', 'Phi2kxt', 'Phi2kzt', 'Phi2zt'] # Electromagnetic spectra must be handled separately as they do not have a species index

    # Initialising plotting
    fig, (ax1) = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("One-dimensional spectra (all simulations)")
    plot_color = plt.cm.rainbow(np.linspace(0, 1, len(simulations.keys())))

    spectrum_count = 0

    # Iterating over similation files 
    for simulation_index, simulation_key in enumerate(simulations.keys()):

        # Selecting simulation
        simulation = simulations[simulation_key]

        # Check whether simulation has same reference species
        species_list, ref_species = extract_species(simulation)
        ref_species_list.append(ref_species)

        if not (ref_species == ref_species_list[0]):
            print("{:} -- WARNING: Inconsistent reference species.".format(simulation_key.name))
            continue

        # Extracting spectra and averaging
        try:
            spectrum_plot = simulation['Spectra'][str(spectrum)]
            plot_range    = simulation['Dimensions'][str(spectrum_range)]
            t_range       = simulation['Dimensions']['time']
            spectrum_count += 1
            
        except:
            print("%s -- WARNING: '%s' not found." %(simulation_key.name, spectrum))
            continue

        it_start          = int(len(t_range) * average_fraction)
        spectrum_time_avg = np.average(spectrum_plot[it_start:], axis = 0)

        # Plotting
        if ref_species == "ions":
            linestyles_dict = {"ions": "solid", "electrons": "dashed"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))
        else:
            linestyles_dict = {"ions": "dashed", "electrons": "solid"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))

        # Plotting time-averaged spectra
        if spectrum in electromagnetic_spectra_list:

            print(r"%s -- maximum at %s = %.4g" % (simulation_key.name, spectrum_range, plot_range[np.argmax(spectrum_time_avg[1:])]))

            plot_label = simulation_key.name

            ax1.plot(plot_range, np.abs(spectrum_time_avg), label=plot_label, linewidth=2, linestyle='solid', marker='o', color=plot_color[simulation_index])
        else:
            for species_index, species in enumerate(species_list):

                print(r"%s -- %s: maximum at %s = %.4g" % (simulation_key.name, species_list[species_index], spectrum_range, plot_range[np.argmax(spectrum_time_avg[species_index, :])]))

                if species_index == 0:
                    plot_label = simulation_key.name
                else:
                    plot_label = None

                if absolute_value:
                    ax1.plot(plot_range, np.abs(spectrum_time_avg[species_index, :]), label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])
                else:
                    ax1.plot(plot_range, spectrum_time_avg[species_index, :], label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])

        # Plotting the spectrum as a function of time
        if time_plots:
            fig, (ax2) = plt.subplots(1, 1)
            fig.canvas.manager.set_window_title("One-dimensional spectra (%s)" % simulation_key.name)

            normalize_time   = matplotlib.colors.Normalize(vmin=0, vmax=t_range.max())
            scalarmappaple = matplotlib.cm.ScalarMappable(norm = normalize_time, cmap = "rainbow")
            plot_colour_it = plt.cm.rainbow(np.linspace(0, 1, len(t_range)))
            colorbar = plt.colorbar(scalarmappaple, ax=ax2)
            colorbar.set_label(r"$t(\sqrt{2}a/v_{\mathrm{th}%s})$" % ref_species[0], labelpad = 15)
            
            # Electromagnetic spectra must be handled separately as they do not have a species index
            if spectrum in electromagnetic_spectra_list:
                for it in range(len(t_range)):
                    if absolute_value:
                        ax2.plot(plot_range, np.abs(spectrum_plot[it, :]), linewidth=2, linestyle='solid', marker='o', color=plot_colour_it[it])
                    else:
                        ax2.plot(plot_range, spectrum_plot[it, :], linewidth=2, linestyle='solid', marker='o', color=plot_colour_it[it])
            else:
                for species_index, species in enumerate(species_list):
                    for it in range(len(t_range)):
                        if absolute_value:
                            ax2.plot(plot_range, np.abs(spectrum_plot[it, species_index, :]), linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_colour_it[it])
                        else:
                            ax2.plot(plot_range, spectrum_plot[it, species_index, :], linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_colour_it[it])
            
            ax2.set_xscale(spectrum_xscale)
            ax2.set_xlabel(spectrum_range_label)
            ax2.set_yscale(spectrum_yscale)
            ax2.set_ylabel(spectrum_label)

    # Checking if there are any spectra to plot
    if spectrum_count == 0:
        print('─' * 100)
        print("Spectrum not found in all simulation files. Returning.")
        print("")
    else:
        # Setting plot options
        ax1.set_xscale(spectrum_xscale)
        ax1.set_xlabel(spectrum_range_label)
        ax1.set_yscale(spectrum_yscale)
        ax1.set_ylabel(spectrum_label)

        # ax1.legend(labelcolor='linecolor', handlelength = 0, title = "Simulations")
        ax1.legend(handlelength = 0, title = "Simulations")

        if spectrum not in electromagnetic_spectra_list:
            print("")
            print("Linestyles:", linestyles_dict)
        print('─' * 100)
        print("Completed.")
        print("")

        plt.show()
    
    return