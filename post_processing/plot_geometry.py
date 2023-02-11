#### Here we make plots of coefficients in gx.eik.out file
# Usage is: python plot_geometry.py geofile.eik.out input_file.in
import csv
import sys
import toml
import numpy as np
from scipy import constants
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
try:
    import pandas as pd
except:
    print("Error: this script requires the pandas python package.") 
    exit(0)
    

##### Plotting parameters
ticklabelsize = 19
label_size = 30
plt.rcParams['xtick.labelsize'] = ticklabelsize
plt.rcParams['ytick.labelsize'] = ticklabelsize
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
colorsrange = ['deepskyblue','coral','darkkhaki','k','mediumvioletred','springgreen','deeppink','lemonchiffon','chocolate','silver','lawngreen','lavender','moccasin','olive','plum','indianred','azure','burlywood','y','blue','green']
linestyleseries = ['-','--','--','--','-','-','--','--','-','-','-','-','-','-','-','-','-','-','-',':',':']

def plot_geometry(fname, input_file_name):

    file_name = fname
    file_name_save = file_name[:-8]
    my_cols = ['A','B','C','D']
    #data_geo_code = pd.read_csv(file_name, encoding = "ISO-8859-1", names=my_cols, lineterminator='\n', engine = 'python')
    try:
        data_geo_code = pd.read_csv(file_name, names=my_cols, engine = 'python')
    except:
        print("Error reading eik.out file. Check formatting.")

    a = data_geo_code.iloc[1]['A']
    splits = a.split()
    nthetagrid = int(splits[0])
    nperiod = int(splits[1])
    ntheta = int(splits[2]) +1
    drhodpsi = float(splits[3])
    rmaj = float(splits[4])
    s_hat = float(splits[5])
    kx_fac = float(splits[6])
    qval = float(splits[7])

    print('Reading {}. \n nperiod = {}, ntheta = {}, drhodpsi = {:.2f}, rmaj = {:.2f}, s_hat = {:.2f}, qval = {:.2f}'.format(file_name,nperiod,ntheta,drhodpsi,rmaj,s_hat,qval))

    gbdrift = np.zeros(ntheta,dtype =float)
    gbdrift0 = np.zeros(ntheta,dtype =float)
    cvdrift = np.zeros(ntheta,dtype =float)
    cvdrift0 = np.zeros(ntheta,dtype =float)
    gradpar = np.zeros(ntheta,dtype =float) # Constant with eq_arc = true
    grho = np.zeros(ntheta,dtype =float)
    thetagrid = np.zeros(ntheta,dtype =float)
    gds2 = np.zeros(ntheta,dtype =float)
    gds21 = np.zeros(ntheta,dtype =float)
    gds22 = np.zeros(ntheta,dtype =float)
    bmag = np.zeros(ntheta,dtype =float)

    for it in np.arange(ntheta):
       
        row_it = 0
        a = data_geo_code.iloc[row_it*(ntheta+1)+it+3]['A']
        splits = a.split()

        gbdrift[it] = float(splits[0])
        gradpar[it] = float(splits[1])
        grho[it] = float(splits[2])
        thetagrid[it] = float(splits[3])

        row_it = 1

        a = data_geo_code.iloc[row_it*(ntheta+1)+it+3]['A']
        splits = a.split()
        cvdrift[it] = float(splits[0])
        gds2[it] = float(splits[1])
        bmag[it] = float(splits[2])

        row_it = 2

        a = data_geo_code.iloc[row_it*(ntheta+1)+it+3]['A']
        splits = a.split()
        gds21[it] = float(splits[0])
        gds22[it] = float(splits[1])

        row_it = 3

        a = data_geo_code.iloc[row_it*(ntheta+1)+it+3]['A']
        splits = a.split()
        cvdrift0[it] = float(splits[0])
        gbdrift0[it] = float(splits[1])

    deltheta = np.diff(thetagrid)

    #### Basic plots

    fig =plt.figure(figsize=(30,15),dpi=300,facecolor='w')
    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242)
    ax3 = plt.subplot(243)
    ax4 = plt.subplot(244)
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246)
    ax7 = plt.subplot(247)
    ax8 = plt.subplot(248)
    cmap = plt.cm.seismic

    ax1.tick_params(axis='x', which='major', pad=5)
    ax1.tick_params(axis='y', which='major', pad=9)
    ax1.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax2.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax3.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax4.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax5.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax6.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax7.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
    ax8.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)

    ax1.set_ylabel('gds2',fontsize=1.1*label_size)
    ax2.set_ylabel('gds21',fontsize=1.1*label_size)
    ax3.set_ylabel('gds22',fontsize=1.1*label_size)
    ax4.set_ylabel('$k_{{\\perp}}/k_y$',fontsize=1.1*label_size)
    ax5.set_ylabel('cvdrift',fontsize=1.1*label_size)
    ax6.set_ylabel('gbdrift',fontsize=1.1*label_size)
    ax7.set_ylabel('cvdrift0',fontsize=1.1*label_size)
    ax8.set_ylabel('$1/|B|$',fontsize=1.1*label_size)

    ax1.plot(thetagrid/np.pi,gds2,ls = '-')
    ax2.plot(thetagrid/np.pi,gds21,ls = '-')
    ax3.plot(thetagrid/np.pi,gds22,ls = '-')
    ax4.plot(thetagrid/np.pi,kperp_over_ky(0,gds2,gds21,gds22),ls = '-')
    ax5.plot(thetagrid/np.pi,cvdrift,ls = '-')
    ax6.plot(thetagrid/np.pi,gbdrift,ls = '-')
    ax7.plot(thetagrid/np.pi,cvdrift0,ls = '-')
    ax8.plot(thetagrid/np.pi,1/bmag,ls = '-')

    #plt.suptitle('$\\theta_0 = 0, \; ${}'.format(file_name_save),fontsize = 1.3*label_size,y = 0.95)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.25)

    plt.savefig(file_name_save + 'geo_dashboard.png',bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()
    plt.close(fig)

    ##### Next, we plot kperp over ky and omega*s/omega_kappa on a contour plot.
    # Creating a finer theta and theta0 grid
    n_theta0_fine = 200
    ntheta_fine = 257
    thetagrid_fine = np.linspace(thetagrid[0],thetagrid[-1],ntheta_fine)
    theta0grid_fine = np.linspace(-np.pi,np.pi,n_theta0_fine)
    thetaindicies = np.arange(ntheta_fine)
    # interpolating geometric quantities on theta grid
    cvdrift_interp = interpolate.interp1d(thetagrid, cvdrift,kind='quadratic')
    cvdrift0_interp = interpolate.interp1d(thetagrid, cvdrift0,kind='quadratic')
    cvdrift_interp_eval = cvdrift_interp(thetagrid_fine)
    cvdrift0_interp_eval = cvdrift0_interp(thetagrid_fine)
    gds2_interp = interpolate.interp1d(thetagrid, gds2,kind='quadratic')
    gds21_interp = interpolate.interp1d(thetagrid, gds21,kind='quadratic')
    gds22_interp = interpolate.interp1d(thetagrid, gds22,kind='quadratic')
    gds2_interp_eval = gds2_interp(thetagrid_fine)
    gds21_interp_eval = gds21_interp(thetagrid_fine)
    gds22_interp_eval = gds22_interp(thetagrid_fine)
    # reading quantities from input file
    #input_file = file_name[:-8] + '.in'
    f = toml.load(input_file_name)
    tprim = np.array(f['species']['tprim'])
    charge = np.array(f['species']['z'])
    mass = np.array(f['species']['mass'])
    temp = np.array(f['species']['temp'])
    species_type = f['species']['type']
    nspec = int(f['Dimensions']['nspecies'])
    # initialising arrays
    kperp_over_ky_array = np.zeros((n_theta0_fine,ntheta_fine),dtype = float)
    omegastar_over_omegakappa_array = np.zeros((n_theta0_fine,ntheta_fine,nspec),dtype = float)
    # setting ky rhos value for evaluation in omega_star and omega_kappa. Ultimately, the ratios kperp_over_ky_array and omegastar_over_omegakappa_array are independent of ky at fixed theta0, so we can choose any value for kyrhorefnow
    kyrhorefnow = 1
    # calculating thermal velocities
    v_thermal_array = np.sqrt(2*temp/mass)
    vt_ref = np.sqrt(2) # reference vthermal
    # reference length
    l_ref = 1

    for spec_it in np.arange(nspec):
        omega_star_s = omstefunction(kyrhorefnow,mass[spec_it],temp[spec_it],v_thermal_array[spec_it],tprim[spec_it],l_ref)    
        for theta0_it in np.arange(n_theta0_fine):
            theta0now = theta0grid_fine[theta0_it]
            if spec_it == 0:
                kperp_over_ky_array[theta0_it,:] = kperp_over_ky_interp(theta0now,gds2_interp_eval,gds21_interp_eval,gds22_interp_eval)
            omegastar_over_omegakappa_array[theta0_it,:,spec_it] = omega_star_s/omegakappa_interp(kyrhorefnow,vt_ref,temp[spec_it],theta0now,cvdrift_interp_eval,cvdrift0_interp_eval,l_ref)

    # calculate kradial = 0 line:
    k_radial_grid_theta0 = k_radial_theta0(theta0grid_fine, kyrhorefnow, ntheta_fine, s_hat, drhodpsi,gds22_interp_eval,gds21_interp_eval)

    #Plotting the perpendicular wavenumber
    fig =plt.figure(figsize=(15,7),dpi=300,facecolor='w')
    ax1 = plt.subplot(111)
    Rmesh, zmesh = np.meshgrid(theta0grid_fine/np.pi,thetagrid_fine/np.pi)
    cmap_inferno = plt.cm.inferno_r
    surf1 = ax1.contourf(zmesh,Rmesh, np.transpose(kperp_over_ky_array[:,:]),24,cmap=cmap_inferno)
    ### Add kradial = 0 line
    surf2 = ax1.contour(zmesh,Rmesh, np.transpose(k_radial_grid_theta0[:,:]),levels = [0],colors = ['crimson'],linestyles = ['--'])
    fmt = {}
    strs = ['$K_x = 0$']
    for l, s in zip(surf2.levels, strs):
        fmt[l] = s
    clabel_fontsize = 22
    ax1.clabel(surf2, surf2.levels, inline=True, fmt=fmt, fontsize=clabel_fontsize)
    ax1.tick_params(axis='x', which='major', pad=5)
    ax1.tick_params(axis='y', which='major', pad=9)
    ax1.set_xlabel('$\\theta / \\pi$',fontsize=1.1*label_size)
    ax1.set_ylabel('$\\theta_0 / \\pi$',fontsize=label_size)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(surf1, cax=cbar_ax)
    cbar_ax.get_yaxis().labelpad = 15
    cbar_ax.set_ylabel('$k_{{\\perp}} / k_y$',fontsize=label_size, rotation=270,labelpad=30)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0+0.025, box.width*0.92, box.height])
    tick_spacing = 0.5
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #plt.suptitle(file_name_save, fontsize = label_size)

    plt.savefig(file_name_save + 'kperp_over_ky.png',bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()
    plt.close(fig)

    #Removing large values
    lowerupper = 20
    omegastar_over_omegakappa_array[omegastar_over_omegakappa_array>lowerupper] = lowerupper
    omegastar_over_omegakappa_array[omegastar_over_omegakappa_array<-lowerupper] = -lowerupper

    #Plotting the magnetic drifts
    for spec_it in np.arange(nspec):

        fig =plt.figure(figsize=(15,7),dpi=300,facecolor='w')
        ax1 = plt.subplot(111)
        Rmesh, zmesh = np.meshgrid(theta0grid_fine/np.pi,thetagrid_fine/np.pi)
        cmap_seis = plt.cm.seismic
        surf1 = ax1.contourf(zmesh,Rmesh, np.transpose(omegastar_over_omegakappa_array[:,:,spec_it]),101,cmap=cmap_seis)

        ax1.tick_params(axis='x', which='major', pad=5)
        ax1.tick_params(axis='y', which='major', pad=9)

        ax1.set_xlabel('$\\theta/\\pi$',fontsize=1.1*label_size)
        ax1.set_ylabel('$\\theta_0/\\pi$',fontsize=label_size)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(surf1, cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel('$\\omega_{{* {} }}^T / \\omega_{{\\kappa {} }}$'.format(species_type[spec_it][0], species_type[spec_it][0]),fontsize=label_size, rotation=270,labelpad=30)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0+0.025, box.width*0.92, box.height])
        #plt.suptitle(file_name_save + species_type[spec_it] + 'drifts', fontsize = label_size)

        plt.savefig(file_name_save + 'omega_star_over_omegakappa' + species_type[spec_it] + '.png',bbox_inches = 'tight', pad_inches = 0.1)
        plt.clf()
        plt.close(fig)

def kperp_over_ky(theta0,gds2,gds21,gds22):
    return np.sqrt(np.abs(gds2+2*theta0*gds21+np.power(theta0,2)*gds22))

def kperp_over_ky_interp(theta0,gds2_interp_eval,gds21_interp_eval,gds22_interp_eval):
    return np.sqrt(np.abs(gds2_interp_eval+2*theta0*gds21_interp_eval+np.power(theta0,2)*gds22_interp_eval))

def omstefunction(kyrhoref,mass_in,temp_in,vt_in,tprim_in,l_ref): # ky rhos * vts * (1/L_{Ts})
    return -0.5*(kyrhoref)*np.sqrt((mass_in)*(temp_in))*(vt_in/l_ref)*tprim_in

def omegakappa_interp(kyrhoref,vt_ref_in,temp_in,theta0,cvdrift_interp_eval,cvdrift0_interp_eval,l_ref): # kperp dot v_curvature
    return np.multiply(-(vt_ref_in/l_ref)*temp_in*kyrhoref/2,np.add(cvdrift_interp_eval, theta0*cvdrift0_interp_eval) ) ;

def k_radial_theta0(theta0_vals, kyrhoref, ntheta, shat,drhodpsi,gds22_interp_eval,gds21_interp_eval):
#### we can obtain the full radial wavenumber with kperp dot (nabla x / |nabla x|) = kx gds22 / sqrt(gds22) + ky (gds21) / sqrt(gds22)
    nakx = len(theta0_vals)
    k_radial_array = np.zeros( (nakx, ntheta), dtype = float)
    for kx_it in np.arange(nakx):
        k_radial_array[kx_it, :] = (1/drhodpsi)*(1/shat)*np.multiply(theta0_vals[kx_it]*kyrhoref*shat,np.sqrt(gds22_interp_eval)) + (1/drhodpsi)*np.divide(np.multiply(kyrhoref,gds21_interp_eval),np.sqrt(gds22_interp_eval))
    return k_radial_array

if __name__ == "__main__":

    print("Plotting various geometric quantities.....")
    fname, inputfilename = sys.argv[1], sys.argv[2]
    try:
        plot_geometry(fname, inputfilename)
    except:
        print('Error... usage: python plot_geometry.py geofile.eik.out input_file.in')
    #plt.show()


