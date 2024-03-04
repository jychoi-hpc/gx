import matplotlib
import sys
from load_files import load_files
from extract_species import extract_species
from plot_spectra_1D import plot_spectra_1D


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
                                spectrum_range=spectrum_range, spectrum_range_label=spectrum_range_label, spectrum_xscale=spectrum_xscale, spectrum_yscale=spectrum_yscale, average_fraction=0.5, absolute_value=False, time=time)
