from netCDF4 import Dataset
from tqdm import tqdm
import pathlib

def load_files(filenames, groups = ['Inputs', 'Geometry', 'Special', 'Spectra', 'Non_zonal', 'Zonal_x', 'Fluxes'], spectra = []):

    # Initialise 
    simulations = {}
    print("")

    # Test for netCDF4 files
    try:
        Dataset(filenames[0], "r", format = "NETCDF4")
        is_netCDF = True
        print("Data has been specified in netCDF4 format.")

    except OSError:
        is_netCDF = False
        print("Data has not been specified in netCDF4 format. Assuming file contains a list of netCDF4 filepaths.")

    # Get filenames 
    if is_netCDF:
        filenames = [pathlib.Path(i) for i in filenames]
    else:
        input_file = pathlib.Path(filenames[0])
        filenames = []
        file_open = open(input_file, "r")
        
        for line in file_open:
            if line[0] == r'#':
                continue
            elif line[0] == '':
                continue
            else:
                filenames.append(pathlib.Path(line.replace('"','').replace("'",'').replace('\n','')))

    # Iterate over files
    print("Loading output file(s).") 
    filenames_iterable = tqdm(filenames, desc=None, leave=True, unit=" files")
    failed_count = 0
    failed_spectra_count = 0

    for filename in filenames_iterable:

        # Initialise
        simulations[filename] = {}

        try:
            # Open netCDF4
            data = Dataset(filename, "r", format = "NETCDF4")
            dimension_names = list(data.variables.keys())
            group_names = list(data.groups.keys())

            # Load dimensions
            simulations[filename]['Dimensions'] = {}
            for key in dimension_names:
                simulations[filename]['Dimensions'][key] = data.variables[key][:]

            # Load data groups
            groups_iterable = tqdm(groups, desc="Loading groups", leave=False, unit=" groups")

            for group in groups_iterable:
                groups_iterable.set_postfix_str("{:}".format(group))
                if group == 'Inputs':
                    simulations[filename][group] = {}
                    subgroup_names = list(data.groups['Inputs'].groups.keys())
                    for subgroup in subgroup_names:
                        if subgroup in {'Controls', 'Species'}:
                            simulations[filename][group][subgroup] = {}
                            keys = list(data.groups[group][subgroup].variables.keys())
                            for key in keys:
                                if key in {'scheme_dum', 'closure_model_dum', 'init_field_dum'}:
                                    simulations[filename][group][subgroup][key] = data.groups[group].groups[subgroup].variables[key].value
                                else:
                                    simulations[filename][group][subgroup][key] = data.groups[group].groups[subgroup].variables[key][:] 
                            subsubgroup_names = list(data.groups[group].groups[subgroup].groups.keys())
                            for subsubgroup in subsubgroup_names:
                                simulations[filename][group][subgroup][subsubgroup] = {}
                                keys = list(data.groups[group][subgroup][subsubgroup].variables.keys())
                                for key in keys:
                                    if key in {'Boltzmann_type_dum', 'forcing_type_dum', 'stir_field_dum'}:
                                        simulations[filename][group][subgroup][subsubgroup][key] = data.groups[group].groups[subgroup].groups[subsubgroup].variables[key].value
                                    else:   
                                        simulations[filename][group][subgroup][subsubgroup][key] = data.groups[group].groups[subgroup].groups[subsubgroup].variables[key][:] 
                        else:
                            simulations[filename][group][subgroup] = {}
                            keys = list(data.groups[group][subgroup].variables.keys())
                            for key in keys:
                                if key in {'boundary_dum', 'restart_from_file_dum', 'restart_to_file_dum', 'source_dum'}:
                                    simulations[filename][group][subgroup][key] = data.groups[group].groups[subgroup].variables[key].value
                                else:
                                    simulations[filename][group][subgroup][key] = data.groups[group].groups[subgroup].variables[key][:] 
                elif group == 'Spectra':
                    simulations[filename][group] = {}
                    for key in spectra:
                        try:
                            simulations[filename][group][key] = data.groups[group].variables[key][:]   
                        except:
                            # tqdm.write("Could not load spectra %s from file: %s" % (key, filename.name))
                            failed_spectra_count += 1
                elif group in group_names:
                    keys = list(data.groups[group].variables.keys())
                    simulations[filename][group] = {}
                    for key in keys:
                        simulations[filename][group][key] = data.groups[group].variables[key][:]   
                else:
                    print("Unable to load group {:}".format(group))

            data.close()

        except:
            tqdm.write("Failed to load file: {:}".format(filename.name))
            failed_count += 1
            del simulations[filename]

        filenames_iterable.set_postfix_str("Failed: files(s) - {:}, spectra - {:}".format(failed_count, failed_spectra_count))

    if failed_count < 1:
        print("Data loaded.")
    else:
        print("Data loaded. Ignoring failed files.")
    
    print("")
    return simulations


