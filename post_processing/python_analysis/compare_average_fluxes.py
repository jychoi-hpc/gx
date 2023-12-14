import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
import collections.abc as collections
import sys
from load_files import load_files
from extract_species import extract_species


def compare_average_fluxes(simuilations, flux, flux_label, defaults = True, average_fraction=0.5, norm_vth=np.sqrt(2), xscale="linear", yscale="log"):

    # print("")

    # Allowing user to specify variables and grouping for the data
    if not defaults:
        variable_str = str(input("Please specify variable to plot against: "))
        grouping_str = str(input("Please specify variable to group the data: "))
    else:
        variable_str = 'q'
        grouping_str = 'T0_prime'

    variable     = None
    grouping     = None

    # The variable names for the 'Species' group are currently different between the input and output file.
    # The following is to allow redundancy in user inputs
    if variable_str in ['mass', 'Mass', 'M']:
        variable_str = 'm'
    if variable_str in ['dens', 'Dens', 'N0']:
        variable_str = 'n0'
    if variable_str in ['temp', 'Temp', 't0']:
        variable_str = 'T0'
    if variable_str in ['tprim', 'Tprim','tprime','Tprime']:
        variable_str = 'T0_prime'
    if variable_str in ['fprim', 'Fprim','fprime','Fprime']:
        variable_str = 'n0_prime'

    if grouping_str in ['mass', 'Mass', 'M']:
        grouping_str = 'm'
    if grouping_str in ['dens', 'Dens', 'N0']:
        grouping_str = 'n0'
    if grouping_str in ['temp', 'Temp', 't0']:
        grouping_str = 'T0'
    if grouping_str in ['tprim', 'Tprim','tprime','Tprime']:
        grouping_str = 'T0_prime'
    if grouping_str in ['fprim', 'Fprim','fprime','Fprime']:
        grouping_str = 'n0_prime'
            
    print("")
    print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []
    data = {}

    # Iterating over similation files to create dictionary 
    for simulation_index, simulation_key in enumerate(simulations.keys()):

        # Selecting simulation
        simulation = simulations[simulation_key]

        # Check whether simulation has same reference species
        species_list, ref_species = extract_species(simulation)
        ref_species_list.append(ref_species)

        if not (ref_species == ref_species_list[0]):
            print("{:} -- WARNING: Inconsistent reference species.".format(simulation_key.name))
            continue

        # Check whether the user variable input is in the output file
        for group in simulation.keys():
            if variable_str in simulation[group].keys():
                variable = simulation[group][variable_str]
            else:
                for subgroup in simulation['Inputs'].keys(): # 'Inputs' is special as it has both subgroups and subsubgroups
                    if subgroup in {'Controls', 'Species'}:
                        if variable_str in simulation['Inputs'][subgroup].keys():
                            variable = simulation['Inputs'][subgroup][variable_str]
                        else:
                            for subsubgroup in simulation['Inputs'][subgroup].keys():
                                try:
                                    if variable_str in simulation['Inputs'][subgroup][subsubgroup].keys():
                                        variable = simulation['Inputs'][subgroup][subsubgroup][variable_str]
                                except:
                                    pass
                    else:
                        if variable_str in simulation['Inputs'][subgroup].keys():
                            variable = simulation['Inputs'][subgroup][variable_str]
        if variable is None:
            print("%s -- WARNING: variable '%s' not found." % (simulation_key.name, variable_str))
            continue

        # Check whether user grouping input is in the output file
        for group in simulation.keys():
            if grouping_str in simulation[group].keys():
                grouping = simulation[group][grouping_str]
            else:
                for subgroup in simulation['Inputs'].keys(): # 'Inputs' is special as it has both subgroups and subsubgroups
                    if subgroup in {'Controls', 'Species'}:
                        if grouping_str in simulation['Inputs'][subgroup].keys():
                            grouping = simulation['Inputs'][subgroup][grouping_str]
                        else:
                            for subsubgroup in simulation['Inputs'][subgroup].keys():
                                try:
                                    if grouping_str in simulation['Inputs'][subgroup][subsubgroup].keys():
                                        grouping = simulation['Inputs'][subgroup][subsubgroup][grouping_str]
                                except:
                                    pass
                    else:
                        if grouping_str in simulation['Inputs'][subgroup].keys():
                            grouping = simulation['Inputs'][subgroup][grouping_str]
        if grouping is None:
            print("%s -- WARNING: grouping '%s' not found. Setting to default (=nan)." % (simulation_key.name, grouping_str))
            grouping = float("nan") # Setting a default value

        # Take variable and grouping from reference species if more than one species is present
        if ref_species == "ions":
            try:
                variable = variable[0]
            except:
                pass
            try:
                grouping = grouping[0]
            except:
                pass
        if ref_species == "electrons":
            try:
                variable = variable[1]
            except:
                pass
            try:
                grouping = grouping[1]
            except:
                pass

        # Casting both the variable and grouping to float for ease of handling 
        variable = float(variable)
        grouping = float(grouping)

        # Extracting flux data
        plot_flux = simulation["Fluxes"][str(flux)]/norm_vth**3
        t_range   = simulation["Dimensions"]["time"] 

        it_start     = int(len(t_range) * average_fraction)
        flux_average = np.average(plot_flux[it_start:], axis=0)
        flux_error   = np.std(plot_flux[it_start:], axis=0) 

        for species_index, species in enumerate(species_list):
            print(r"%s -- %s_%s/%s_gB%s = %.5g +/- %.5g" % (simulation_key.name, flux_label, species[0], flux_label, ref_species[0], flux_average[species_index], flux_error[species_index]))

        # Sort data by grouping
        if grouping is not None:
            if grouping not in list(data.keys()):
                data[grouping] = {}
                data[grouping] = {variable_str: [variable], "flux_average": [flux_average], "flux_error": [flux_error]}
            elif grouping in list(data.keys()):
                data[grouping][variable_str].append(variable)
                data[grouping]["flux_average"].append(flux_average)
                data[grouping]["flux_error"].append(flux_error)
        else:
            if len(data) == 0:
                data = {variable_str: [variable], "flux_average": [flux_average], "flux_error": [flux_error]} 
            else:
                data[variable_str].append(variable)
                data["flux_average"].append(flux_average)
                data["flux_error"].append(flux_error)

    # Sorting data be in ascending order within the groupings
    if grouping is not None:
        groupings = list(set(list(data.keys())))  # Extract unique groupings and sort
        groupings.sort() 
        for grouping in groupings:
            sorted_indices = np.argsort(data[grouping][variable_str])
            for key in data[grouping].keys():
                data[grouping][key] = np.array(data[grouping][key])[sorted_indices]

    if variable is not None:

        # Plotting
        if ref_species == "ions":
            markers_dict = {"ions": "o", "electrons": "s"} | dict(zip(species_list[2:], list(islice(cycle(["^", "v"]), len(species_list[2:])))))
        else:
            markers_dict = {"ions": "s", "electrons": "o"} | dict(zip(species_list[2:], list(islice(cycle(["^", "v"]), len(species_list[2:])))))

        fig, (ax1) = plt.subplots(1, 1)
        plot_color = plt.cm.rainbow(np.linspace(0, 1, len(data.keys())))

        for species_index, species in enumerate(species_list):
            if grouping is not None:
                for grouping_index, grouping in enumerate(groupings):
                    ax1.errorbar(data[grouping][variable_str], data[grouping]["flux_average"][:, species_index], label=r"%s = %.1f" % (grouping_str, grouping), \
                                marker=markers_dict[species], markersize=10, color=plot_color[grouping_index], linestyle="solid", linewidth=2,  xerr=None, yerr=data[grouping]['flux_error'][:, species_index], elinewidth=3)
            else:
                ax1.errorbar(data[variable_str], data["flux_average"], label=None, \
                            marker=markers_dict[species], markersize=10, color=plot_color[0], linestyle="solid", linewidth=2,  xerr=None, yerr=data["flux_error"], elinewidth=3)

        ax1.set_xscale(xscale)
        ax1.set_xlabel(r"%s" % (variable_str))
        ax1.set_yscale(yscale)
        ax1.set_ylabel(r"$%s/%s_{\mathrm{gB}%s}$" % (flux_label, flux_label, ref_species[0]))

        ax1.legend(labelcolor='linecolor', title="Simulations", loc="lower right")

        print("")
        print("Markers:", markers_dict)

        print('─' * 100)
        print("Completed.")
        print("")

        plt.show()
    
    else:
        print('─' * 100)
        print("Variable '%s' not found in any simulation. Aborting." % variable_str)
        print("")

    return


if __name__ == "__main__":

    # Interpreting command line input
    filenames = sys.argv[1:]

    if "latex" in filenames:
        filenames.remove("latex")

        font = {'family' : 'serif',
        'serif'  : ['Computer Modern Roman'],
        'weight' : 'bold',
        'size'   : 38} # Use 40 with latex for papers, 18 otherwise

        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['axes.labelpad']='20'
        
        print("")
        print("Using LaTeX")

    defaults = True

    if "user_input" in filenames:
        filenames.remove("user_input")
        defaults = False

    fluxes       = ["qflux", "pflux"]
    fluxes_label = ["Q", "\Gamma"]

    fluxes_plot       = []
    fluxes_label_plot = []

    for flux_index, flux in enumerate(fluxes):
        if flux in filenames:
            filenames.remove(flux)
            fluxes_plot.append(flux)
            fluxes_label_plot.append(fluxes_label[flux_index])

    if len(fluxes_plot) == 0:
        print("")
        print("Please specify one (or more) of the following as a command-line input:")
        print(fluxes)
        print("")

        exit()

    else:
        simulations = load_files(filenames)

        for flux_index, flux in enumerate(fluxes_plot):
            compare_average_fluxes(simulations, flux=fluxes_plot[flux_index], flux_label=fluxes_label_plot[flux_index], defaults=defaults, yscale="linear", average_fraction=0.6)