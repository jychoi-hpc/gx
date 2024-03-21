import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
import sys
from load_files import load_files
from extract_species import extract_species


def plot_spectra_1D(simulations, spectrum, spectrum_label, spectrum_range, spectrum_range_label, spectrum_xscale, spectrum_yscale, average_fraction=0.5, absolute_value=False, time=False, modes=False):

    plt.close("all")

    # print("")
    # print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []
    electromagnetic_spectra_list = ['Phi2kyt', 'Phi2kxt', 'Phi2kzt', 'Phi2zt'] # Electromagnetic spectra must be handled separately as they do not have a species index

    # Initialising plotting
    fig, (ax) = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("One-dimensional spectra (averaged)")
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

            ax.plot(plot_range, np.abs(spectrum_time_avg), label=plot_label, linewidth=2, linestyle='solid', marker='o', color=plot_color[simulation_index])
        else:
            for species_index, species in enumerate(species_list):

                print(r"%s -- %s: maximum at %s = %.4g" % (simulation_key.name, species_list[species_index], spectrum_range, plot_range[np.argmax(spectrum_time_avg[species_index, :])]))

                if species_index == 0:
                    plot_label = simulation_key.name
                else:
                    plot_label = None

                if absolute_value:
                    ax.plot(plot_range, np.abs(spectrum_time_avg[species_index, :]), label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])
                else:
                    ax.plot(plot_range, spectrum_time_avg[species_index, :], label=plot_label, linewidth=2, linestyle=linestyles_dict[species], marker='o', color=plot_color[simulation_index])

        # Plotting the spectrum as a function of time
        if time:

            fig, (ax_time) = plt.subplots(1, 1)
            fig.canvas.manager.set_window_title("One-dimensional spectra (%s)" % simulation_key.name)

            norm     = matplotlib.colors.Normalize(vmin=0, vmax=t_range.max())
            colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap = "rainbow"), ax=ax_time)
            colorbar.set_label(r"$t(\sqrt{2}a/v_{\mathrm{th}%s})$" % ref_species[0], labelpad = 15)

            interval_scale = 1
            it_count       = len(t_range)//interval_scale
            plot_color     = plt.cm.rainbow(np.linspace(0, 1, num=it_count, endpoint=False))
            
            # Electromagnetic spectra must be handled separately as they do not have a species index
            if spectrum in electromagnetic_spectra_list:
                for it_index, it in enumerate(np.linspace(0, len(t_range), num=it_count, dtype=int, endpoint=False)):
                    if absolute_value:
                        ax_time.plot(plot_range, np.abs(spectrum_plot[it, :]), linewidth=2, linestyle='solid', marker='o', color=plot_color[it_index])
                    else:
                        ax_time.plot(plot_range, spectrum_plot[it, :], linewidth=2, linestyle='solid', marker='o', color=plot_color[it_index])
            else:
                for species_index, species in enumerate(species_list):
                    for it_index, it in enumerate(np.linspace(0, len(t_range), num=it_count, dtype=int, endpoint=False)):
                        if absolute_value:
                            ax_time.plot(plot_range, np.abs(spectrum_plot[it, species_index, :]), linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_color[it_index])
                        else:
                            ax_time.plot(plot_range, spectrum_plot[it, species_index, :], linewidth=2, linestyle=linestyles_dict[species], marker = 'o', color=plot_color[it_index])
            
            ax_time.set_xscale(spectrum_xscale)
            ax_time.set_xlabel(spectrum_range_label)
            ax_time.set_yscale(spectrum_yscale)
            ax_time.set_ylabel(spectrum_label)

        # Plotting the time-evolution of the individual modes
        if modes:

            fig, (ax_modes) = plt.subplots(1, 1)
            fig.canvas.manager.set_window_title("One-dimensional spectra (%s)" % simulation_key.name)

            k_start = plot_range[1]
            k_stop  = plot_range[-1]
            ik_start = np.absolute(plot_range - k_start).argmin()
            ik_stop  = np.absolute(plot_range - k_stop).argmin()

            norm     = matplotlib.colors.Normalize(vmin=plot_range[1], vmax=plot_range[-1]) # Colours are normalised to the full range of possible modes, and we select a subset.
            colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="rainbow"), ax=ax_modes)
            colorbar.set_label(spectrum_range_label, labelpad = 15)

            interval_scale = 1
            ik_count       = len(plot_range[ik_start:ik_stop])//interval_scale
            plot_color     = plt.cm.rainbow(np.linspace(0, 1, num=len(plot_range), endpoint=False))

            # Electromagnetic spectra must be handled separately as they do not have a species index
            if spectrum in electromagnetic_spectra_list:
                for ik_index, ik in enumerate(np.linspace(ik_start, (ik_stop), num=ik_count, dtype=int, endpoint=False)):
                    if absolute_value:
                        ax_modes.plot(t_range, np.abs(spectrum_plot[:, ik]), linewidth=2, linestyle='solid', marker='', color=plot_color[ik])
                    else:
                        ax_modes.plot(t_range, spectrum_plot[:, ik], linewidth=2, linestyle='solid', marker='', color=plot_color[ik])
            else:
                for species_index, species in enumerate(species_list):
                    for ik_index, ik in enumerate(np.linspace(ik_start, (ik_stop), num=ik_count, dtype=int, endpoint=False)):
                        if absolute_value:
                            ax_modes.plot(t_range, np.abs(spectrum_plot[:, species_index, ik]), linewidth=2, linestyle=linestyles_dict[species], marker='', color=plot_color[ik])
                        else:
                            ax_modes.plot(t_range, spectrum_plot[:, species_index, ik], linewidth=2, linestyle=linestyles_dict[species], marker='', color=plot_color[ik])
            
            ax_modes.set_xscale("linear")
            ax_modes.set_xlabel(r"$t(\sqrt{2}a/v_{\mathrm{th}%s})$" % ref_species[0])
            ax_modes.set_yscale(spectrum_yscale)
            ax_modes.set_ylabel(spectrum_label)
            

    # Checking if there are any spectra to plot
    if spectrum_count == 0:
        print('─' * 100)
        print("Spectrum not found in all simulation files. Returning.")
        print("")
    else:
        # Setting plot options
        ax.set_xscale(spectrum_xscale)
        ax.set_xlabel(spectrum_range_label)
        ax.set_yscale(spectrum_yscale)
        ax.set_ylabel(spectrum_label)

        # ax.legend(labelcolor='linecolor', handlelength = 0, title = "Simulations")
        ax.legend(handlelength = 0, title = "Simulations")

        if spectrum not in electromagnetic_spectra_list:
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

    # Setting plotting options
    time = False
    if "time" in filenames:
        filenames.remove("time")
        time = True

    modes = False
    if "modes" in filenames:
        filenames.remove("modes")
        modes = True

    # Lists of spectra
    spectra_kx       = ["Wkxst", "Pkxst", "Qkxst", "Gamkxst", "Phi2kxt"]
    spectra_label_kx = [r"$W_s(k_x)$", r"$[1-\Gamma_0(b_s)]|\phi(k_x)|^2$", r"$Q_s(k_x)$", r"$\Gamma_s(k_x)$", r"$|\phi(k_x)|^2$"]

    spectra_ky       = ["Wkyst", "Pkyst", "Qkyst", "Gamkyst", "Phi2kyt"]
    spectra_label_ky = [r"$W_s(k_y)$", r"$[1-\Gamma_0(b_s)]|\phi(k_y)|^2$", r"$Q_s(k_y)$", r"$\Gamma_s(k_y)$", r"$|\phi(k_y)|^2$"]

    spectra_kz       = ["Wkzst", "Pkzst", "Qkzst", "Gamkzst", "Phi2kzt"]
    spectra_label_kz = [r"$W_s(k_z)$", r"$[1-\Gamma_0(b_s)]|\phi|^2(k_z)$", r"$Q_s(k_z)$", r"$\Gamma_s(k_z)$", r"$|\phi(k_z)|^2$"]

    spectra_theta       = ["Wzst" , "Pzst", "Qzst", "Gamzst", "Phi2zt"]
    spectra_label_theta = [r"$W_s(\theta)$", r"$[1-\Gamma_0(b_s)]|\phi(\theta)|^2$", r"$Q_s(\theta)$", r"$\Gamma_s(\theta)$", r"$|\phi(\theta)|^2$"]

    spectra       = spectra_kx + spectra_ky + spectra_kz + spectra_theta
    spectra_label = spectra_label_kx + spectra_label_ky + spectra_label_kz + spectra_label_theta

    spectra_plot       = []
    spectra_label_plot = []

    # Appending user inputs
    if "all" in filenames:
        filenames.remove("all")
        spectra_plot = spectra
        spectra_label_plot = spectra_label

    for spectrum_index, spectrum in enumerate(spectra):
        if spectrum in filenames:
            filenames.remove(spectrum)
            spectra_plot.append(spectrum)
            spectra_label_plot.append(spectra_label[spectrum_index])

    if len(spectra_plot) == 0:
        print("")
        print("Please specify 'all', or one (or more) of the following as a command-line input:")
        print(spectra_kx)
        print(spectra_ky)
        print(spectra_kz)
        print(spectra_theta)
        print("")

        exit()

    else:
        simulations = load_files(filenames, groups=['Inputs', 'Spectra'], spectra=spectra_plot)

        for spectrum_index, spectrum in enumerate(spectra_plot):
            print("Plotting", spectrum)

            if spectrum in spectra_kx:
                spectrum_range       = 'kx'
                spectrum_range_label = r'$k_x$'
                spectrum_xscale      = "symlog"
                spectrum_yscale      = "log"
                
            elif spectrum in spectra_ky:
                spectrum_range       = 'ky'
                spectrum_range_label = r'$k_y$'
                spectrum_xscale      = "log"
                spectrum_yscale      = "log"

            elif spectrum in spectra_kz:
                spectrum_range       = 'kz'
                spectrum_range_label = r'$k_z$'
                spectrum_xscale      = "log"
                spectrum_yscale      = "log"

            elif spectrum in spectra_theta:
                spectrum_range       = 'theta'
                spectrum_range_label = r'$\theta$'
                spectrum_xscale      = "linear"
                spectrum_yscale      = "linear"
                
            plot_spectra_1D(simulations, spectrum=spectra_plot[spectrum_index], spectrum_label=spectra_label_plot[spectrum_index], \
                                spectrum_range=spectrum_range, spectrum_range_label=spectrum_range_label, spectrum_xscale=spectrum_xscale, spectrum_yscale=spectrum_yscale, average_fraction=0.5, absolute_value=False, time=time, modes=modes)
