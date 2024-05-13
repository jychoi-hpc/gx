"""
Loads data from one (or more) GX output files into a Python dictionary to be used by post-processing scripts.
These are specified by providing a set of filepaths pointing to the .out.nc file(s), either directly in the command line 
or using the path to a .txt file containing the filepath(s) on each line. 
"""

from netCDF4 import Dataset
from tqdm import tqdm
import pathlib


def load_out_nc(simulation, filename, diagnostics):
    """
    Loads data from .out.nc file. The groups 'Dimensions', 'Grids', 'Geometry', and 'Inputs' are loaded by default.
    Only specified diagnostics are loaded due to their potential size. 
    """

    # Open .out.nc file
    data = Dataset(filename, 'r', format='NETCDF4')

    # Load dimensions
    simulation['Dimensions'] = {}
    for key in list(data.variables.keys()):
        simulation['Dimensions'][key] = data.variables[key]

    # Load grids
    simulation['Grids'] = {}
    for key in list(data.groups['Grids'].variables.keys()):
        simulation['Grids'][key] = data.groups['Grids'].variables[key]

    # Load inputs
    simulation['Inputs'] = {}

    simulation['Inputs']['Geometry'] = {}
    for key in list(data.groups['Inputs'].variables.keys()):
        simulation['Inputs']['Geometry'][key] = data.groups['Inputs'].variables[key]

    for subgroup in list(data.groups['Inputs'].groups.keys()):
        simulation['Inputs'][subgroup] = {}
        if subgroup in {'Controls', 'Species'}:
            for key in list(data.groups['Inputs'].groups[subgroup].variables.keys()):
                simulation['Inputs'][subgroup][key] = data.groups['Inputs'].groups[subgroup].variables[key]
            for subsubgroup in list(data.groups['Inputs'].groups[subgroup].groups.keys()):
                simulation['Inputs'][subgroup][subsubgroup] = {}
                for key in list(data.groups['Inputs'].groups[subgroup].groups[subsubgroup].variables.keys()):
                    simulation['Inputs'][subgroup][subsubgroup][key] = data.groups['Inputs'].groups[subgroup].groups[subsubgroup].variables[key]
        else:
            for key in list(data.groups['Inputs'].groups[subgroup].variables.keys()):
                simulation['Inputs'][subgroup][key] = data.groups['Inputs'].groups[subgroup].variables[key]

    # Load diagnostics
    if len(diagnostics) > 0:
        simulation['Diagnostics'] = {}
        for diagnostic in diagnostics:
            if diagnostic in list(data.groups['Diagnostics'].variables.keys()):
                simulation['Diagnostics'][diagnostic] = data.groups['Diagnostics'].variables[diagnostic]
            else:
                tqdm.write("Diagnostic '%s' not found in %s" % (diagnostic, filename))

    data.close()

    return


def load_big_nc():

    return


def load_files(filenames, diagnostics=[]):
    """
    Main loading function. 
    """

    # Initialising dictionary
    simulations = {}
    print("")

    # Test file type and get filenames
    if pathlib.Path(filenames[0]).suffix == '.txt':

        input_file = pathlib.Path(filenames[0])
        filenames  = []
        file_open  = open(input_file, 'r')

        for line in file_open:
            if line[0] == r'#':
                continue
            elif line[0] == '':
                continue
            else:
                filenames.append(pathlib.Path(line.replace('"','').replace("'",'').replace('\n','')))
    else:
        filenames = [pathlib.Path(i) for i in filenames]

    # Iterate over files
    print("Loading output file(s).")
    filenames_iterable = tqdm(filenames, desc=None, leave=True, unit=" files")
    failed_files_count = 0

    for filename in filenames_iterable:

        # Initialise
        simulations[filename] = {}
        
        # try:
        # Loading data from .out.nc file
        load_out_nc(simulations[filename], filename, diagnostics)

        # except:
        #     tqdm.write("Failed to load file: %s" % (filename))
        #     failed_files_count += 1
        #     del simulations[filename]

        # filenames_iterable.set_postfix_str("Failed: files(s) - %s" % (failed_files_count))

    # Error messages
    if failed_files_count < 1:
        print("Data loaded.")

    elif failed_files_count == len(filenames):
        print("No data could be loaded. Aborting.")
        print("")
        exit()
    else:
        print("Data loaded. Ignoring failed files.")
    
    # Returning dictionary
    print("")
    return simulations


if __name__ == "__main__":

    import sys
    import time
    import numpy as np

    avg = []

    for i in range(1):

        t0 = time.time()
        # Interpreting command line input
        filenames = sys.argv[1:]

        simulations = load_files(filenames, diagnostics=['HeatFluxBpar_zst', 'blah'])

        # print(simulations.keys())

        sim = simulations[pathlib.Path(filenames[0])]

        # for key in sim['Grids'].keys():
        #     print(key, type(sim['Grids'][key][:]), sim['Grids'][key][:])

        # for key in sim['Geometry'].keys():
        #     print(key, sim['Geometry'][key][:])

        # print(simulations)

        t1 = time.time()

        dt = t1-t0

        avg.append(dt)

    print(np.average(avg))

