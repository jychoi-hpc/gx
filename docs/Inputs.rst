Inputs
======

GX is currently set up to be run from input files. The main input file has a user-supplied name,
such as ``a01`` or ``scan/a01``, to which the suffix **.in** should be appended. In these examples, 
the input file names are ``a01.in`` and ``scan/a01.in``. In the first case, the input file is in
the current directory. In the second case, the input file is in a directory called ``scan``,
which itself is located in the current directory.

Important: If you are using an input file in another directory, and if you are also referring to additional
files (perhaps for a restart, perhaps to specify geometric information, etc) then the file names
for the additional files must also include the longer path.


The input file
--------------
*There are two standards for the input file. One is deprecated and is not documented here.
If you are exploring src/parameters.cu and find yourself confused by the logic there, you are
almost certainly looking at obscure coding that supports the old standard. That obscure coding
is itself deprecated.*

The main GX input file is parsed with `toml <https://github.com/ToruNiina/toml11>`_.

The toml standard allows one to group related input variables. There are multiple
ways to specify such groupings. Since there are sample input files provided
with GX, here we will pick one particular syntactical pattern. Users are free
to use anything that is compatible with the `toml <https://github.com/ToruNiina/toml11>`_ standard.

There are two input variables that are not part of any group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - *none*
     - debug
     - If true, print debug information to standard out
     - **false**
   * - *none*
     - repeat 
     - If true, repeat a previous run, using the information in the
       output file from that run. **Presently not available**
     - **false**

The grid sizes are specied in the **Dimensions** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Dimensions**
     - nx
     - The number of grid points used in the *x* direction.
     - **4**
   * - **Dimensions**
     - ny
     - The number of grid points used in the *y* direction.
     - **32**
   * - **Dimensions**
     - ntheta
     - The number of grid points used in the *z* direction.
     - **32**
   * - **Dimensions**
     - nhermite
     - The number of Hermite basis functions used.
     - **4**
   * - **Dimensions**
     - nlaguerre
     - The number of Laguerre basis functions used.
     - **2**
   * - **Dimensions**
     - nspecies
     - The number of kinetic species used.
     - **1**
   * - **Dimensions**
     - nperiod
     - The number of poloidal turns used. 
     - **1**

The physical extent of the simulation domain is specied in the **Domain** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Domain**
     - x0
     - The extent of the radial domain is 2 pi x0. 
     - **10.0**
   * - **Domain**
     - y0
     - The extent of the binormal domain is 2 pi y0. 
     - **10.0**
   * - **Domain**
     - jtwist
     - The twist-and-shift boundary condition is controlled by jtwist. Typically,
       you should set jtwist to be an integer close to 2 pi / s_hat. 
     - **-1**
   * - **Domain**
     - boundary
     - Two options are possible: **"periodic"** or **"linked"**
     - **"linked"**
   * - **Domain**
     - ExBshear
     - **Not yet implemented.** Set to **true** to include equilibrium ExB shear. 
     - **false**
   * - **Domain**
     - g_exb
     - **Not yet implemented.** Sets the equilibrium ExB shearing rate. 
     - **0.0**

The length of simulated time and the timestep are set in the **Time** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Time**
     - dt
     - The maximum timestep allowed.
     - **0.05**
   * - **Time**
     - nstep
     - The number of timesteps to take. 
     - **10000**
   * - **Time**
     - nwrite
     - Write time-dependent information every nwrite timesteps. 
     - **1000**
   * - **Time**
     - navg
     - **Not yet implemented.** Average time-dependent information over navg timesteps.
     - **10**
   * - **Time**
     - nsave
     - **Not yet implemented.** Write a restart file every nsave timesteps. 
     - **2000000**

The properties for each kinetic species are specified in the **species** group.
For the most part, there are mostly no default values provided:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **species**
     - z
     - The charge
     -
   * - **species**
     - mass
     - The mass
     -
   * - **species**
     - dens
     - The density
     -
   * - **species**
     - temp
     - The temperature
     - **1.0**
   * - **species**
     - tprim
     - L/LT
     -
   * - **species**
     - fprim
     - L/Ln
     - 
   * - **species**
     - uprim
     - L/Lu
     - **0.0**
   * - **species**
     - vnewk
     - The collision frequency
     - **0.0**
   * - **species**
     - type
     - The type of species, such as ion or electron
     - **"ion"**
    
The **Controls** group contains switches and variables that determine both the physics model
and various numerical parameters.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Controls**
     - nonlinear_mode
     - Set to true to include nonlinear terms in the equations. 
     - **false**
   * - **Controls**
     - scheme
     - This string variable chooses the time-stepping scheme to be used. 
     - **"sspx3"**
   * - **Controls**
     - stages
     - The number of Runge-Kutta stages to be used for certain time-stepping schemes. Not relevant
       for most choices of **scheme**. 
     - **10**    
   * - **Controls**
     - cfl
     - For nonlinear runs, the maximum timestep allowed is proportional to **cfl**.
     - **1.0**    
   * - **Controls**
     - init_field
     - The initial perturbation is applied to this component of the distribution function.
     - **"density"**    
   * - **Controls**
     - init_amp
     - The initial perturbation has this amplitude. 
     - **1.0e-5**    
   * - **Controls**
     - kpar_init
     - The initial perturbation has this parallel wavenumber.
     - **0.0**
   * - **Controls**
     - closure_model
     - This string variable determines which (if any) closure model to use.
     - **"none"**
   * - **Controls**
     - fphi
     - Set fphi to 1.0 to include Phi perturbations in the calculation. 
     - **1.0** 
   * - **Controls**
     - fapar
     - **Not yet implemented.** Set fapar to 1.0 to include perturbations of the
       parallel component of the vector potential in the calculation. 
     - **0.0**
   * - **Controls**
     - fbpar
     - **Not yet implemented.** Set fbpar to 1.0 to include perturbations of the
       magnetic field strength in the calculation. 
     - **0.0**
   * - **Controls**
     - HB_hyper
     - If true, use the Hammett-Belli hyperdiffusivity model 
     - **false**
   * - **Controls**
     - D_HB
     - If HB_hyper is true, sets the strength of the H-B hyperdiffusivity model.
     - **1.0**
   * - **Controls**
     - w_osc
     - If HB_hyper is true, sets the frequency parameter in the H-B model.
     - **0.0**
   * - **Controls**
     - p_HB
     - If HB_hyper is true, sets the exponent for the H-B model,
       where 2 corresponds to the fourth power of k.
     - **2**

       
To continue a previous run, use the  **Restart** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Restart**
     - restart
     - Set to true to continue from a previous run.
     - **false**
   * - **Restart**
     - save_for_restart
     - Set to true to write information needed for a future restart. File is written at the end of the run. 
     - **true**
   * - **Restart**
     - restart_to_file
     - Filename to use for restart information for the present run.       
     - **"newsave.nc"**
   * - **Restart**
     - restart_from_file
     - Filename for a file written from a previous run, to be used as the basis for continuation (restart).
     - **"oldsave.nc"**
   * - **Restart**
     - scale
     - Multiply all variables in the restart variable by a factor of **scale**. 
     - **1.0**

The **Diagnostics** group controls the kinds of diagnostic information that is produced.
Diagnostic information is written in NetCDF format, to a file that uses the input file name
(without the **.in** suffix) with the **.nc** suffix appended.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Diagnostics**
     - omega
     - Write instantaneous estimates of the complex frequency for each Fourier component of the electrostatic potential.
     - **false**
   * - **Diagnostics**
     - free_energy
     - Write the total free energy (integrated over the phase-space domain and summed over species) as a function of time. 
     - **true**
   * - **Diagnostics**
     - fluxes
     - Write the turbulent fluxes for each species. 
     - **false**
   * - **Diagnostics**
     - all_zonal_scalars
     - Write quantities such as the RMS value of the zonal component of the ExB velocity as function of time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - avg_zvE
     - Write the RMS value of the zonal component of ExB velocity as a function time.
     - **false**
   * - **Diagnostics**
     - avg_zkvE
     - Write the RMS value of the zonal component of the shear of the ExB velocity as a function time.
     - **false**
   * - **Diagnostics**
     - avg_zkden
     - Write the RMS value of the zonal component of the guiding center
       radial density gradient as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkUpar
     - Write the RMS value of the zonal component of the guiding center
       parallel velocity as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkTpar
     - Write the RMS value of the zonal component of the guiding center
       parallel temperature as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkTperp
     - Write the RMS value of the zonal component of the guiding center
       perpendicular temperature as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkqpar
     - Write the RMS value of the zonal component of the guiding center
       parallel-parallel heat flux as a function of time.
     - **false**
   * - **Diagnostics**
     - all_zonal
     - Write quantities such as the zonal component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - vE
     - Write the zonal component of ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kvE
     - Write the zonal component of the shear of the ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kden
     - Write the zonal component of the radial gradient of the guiding center
       density as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kUpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kTpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kTperp
     - Write the zonal component of the radial gradient of the guiding center
       perpendicular temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kqpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel-parallel heat flux as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - all_non_zonal
     - Write quantities such as the non-zonal *y*-component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - xyvEx
     - Write the *x*-component of the ExB velocity as a function of *x*, *y*, and time.
     - **false**
   * - **Diagnostics**
     - xyvEy
     - Write the non-zonal *y*-component of the ExB velocity as a function of *x*, *y*, and time.
     - **false**
   * - **Diagnostics**
     - xykvE
     - Write the non-zonal part of the shear of the *y*-component of the ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyden
     - Write the non-zonal component of the guiding center density as a function of *x* and time.       
     - **false**
   * - **Diagnostics**
     - xyUpar
     - Write the non-zonal component of the guiding center parallel velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyTpar
     - Write the non-zonal component of the guiding center parallel temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyTperp
     - Write the non-zonal component of the guiding center perpendicular temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyqpar
     - Write the non-zonal component of the guiding center parallel-parallel heat flux as a function of *x* and time.
     - **false**

The **Expert** group is for expert users: 

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Expert**
     - i_share
     - An integer related to the shared memory block used for the inner loop of the linear solver.
     - **8**
   * - **Expert**
     - nreal 
     - Enforce the reality condition every nreal timesteps. 
     - **1**
   * - **Expert**
     - init_single
     - Only initialize a single Fourier mode if true
     - **false**
   * - **Expert**
     - ikx_single
     - Index of the kx mode to be initialized if init_single is true
     - **0**
   * - **Expert**
     - iky_single
     - Index of the ky mode to be initialized if init_single is true
     - **1**
   * - **Expert**
     - eqfix
     - Do not evolve some particular Fourier harmonic
     - **false**
   * - **Expert**
     - ikx_fixed
     - Index of the kx mode to be fixed in time if eqfix is true
     - **-1**
   * - **Expert**
     - iky_fixed
     - Index of the ky mode to be fixed in time if eqfix is true
     - **-1**
   * - **Expert**
     - secondary
     - Set things up for a secondary instability calculation
     - **false**
   * - **Expert**
     - phi_ext
     - Value of phi to use for a Rosenbluth-Hinton test
     - **0.0**
   * - **Expert**
     - source
     - Used to specify various kinds of tests
     - **"default"**
   * - **Expert**
     - tp_t0
     - The time at which to start changing tprim
     - **-1.0**
   * - **Expert**
     - tp_tf
     - The time at which to stop changing tprim
     - **-1.0**
   * - **Expert**
     - tprim0
     - The value of tprim to start with 
     - **-1.0**
   * - **Expert**
     - tprimf
     - The value of tprim to end with 
     - **-1.0**
       
The size and resolution of the simulation domain can be changed in
certain ways in a restarted run. These are new options that have not
been used much. The group is **Resize**:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Resize**
     - domain_change
     - Allow the functionality of this group to be used if domain_change is true
     - **false**
   * - **Resize**
     - x0_mult
     - Multiply Lx by x0_mult. Must be an integer >= 1.
     - **1**
   * - **Resize**
     - y0_mult
     - Multiply Ly by y0_mult. Must be an integer >= 1.
     - **1**
   * - **Resize**
     - z0_mult
     - Multiply the parallel box length by z0_mult. Must be an integer >= 1. Only valid for unsheared slab for now.
     - **1**
   * - **Resize**
     - nx_mult
     - Multiply the number of grid points in the x-direction by nx_mult. Must be an integer >= 1. 
     - **1**
   * - **Resize**
     - ny_mult
     - Multiply the number of grid points in the y-direction by ny_mult. Must be an integer >= 1. 
     - **1**
   * - **Resize**
     - ntheta_mult
     - Multiply the number of grid points in the z-direction by ntheta_mult. Must be an integer >= 1. 
     - **1**
   * - **Resize**
     - nm_add
     - Add nm_add Hermite moments. Must be integer, can be positive, negative or zero. 
     - **0**
   * - **Resize**
     - nl_add
     - Add nl_add Laguerre moments. Must be integer, can be positive, negative or zero. 
     - **0**
   * - **Resize**
     - ns_add
     - Add ns_add species.  Must be integer >= 0.
     - **0**

Add forcing with the **Forcing** group. Not generally implemented. 

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Forcing**
     - forcing_type
     - Picks among the forcing options
     - **"Kz"**
   * - **Forcing**
     - stir_field
     - Determines which moment of the GK equation is forced
     - **"density"**
   * - **Forcing**
     - forcing_amp
     - Amplitude of the forcing
     - **1.0**
   * - **Forcing**
     - forcing_index
     - Index of the forcing
     - **1**
   * - **Forcing**
     - no_fields
     - Turn off the field terms in the GK equation if this is true
     - **false**

One component of the plasma can be assumed to have a Boltzmann response. This is controlled with
the **Boltzmann** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Boltzmann**
     - add_Boltzmann_species
     - Include a species with a Boltzmann response if true
     - **false**
   * - **Boltzmann**
     - Boltzmann_type
     - Choose either "electrons" or "ions"
     - **"electrons"**
   * - **Boltzmann**
     - tau_fac
     - Set the value of tau for the Boltzmann species.
       Actual default value is -1.0, but this is for obscure reasons. 
       Use 1.0 as the default value and always choose a positive value. 
     - **-1.0**
       
The geometry of the simulation domain is controlled through the **Geometry** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Geometry**
     - igeo 
     - Integer. To get an analytic form of the equilibrium, use igeo = 0.
       To read the geometric information from a file, use igeo = 1. No other options are implemented for now. 
     - **0**
   * - **Geometry**
     - geofilename
     - If igeo = 1, the geometric information is read from geofilename. 
     - **"eik.out"**
   * - **Geometry**
     - slab
     - If true, and if igeo = 0, the geometry is that of a slab.
     - **false**
   * - **Geometry**
     - const_curv
     - If true, and if igeo = 0, the curvature is assumed to a constant, as in a Z-pinch. 
     - **false**
   * - **Geometry**
     - drhodpsi
     - Not used.
     - **1.0**
   * - **Geometry**
     - kxfac
     - Not used.
     - **1.0**
   * - **Geometry**
     - Rmaj
     - If igeo = 0, Rmaj is the ratio of the major radius to the equilibrium-scale reference length.
       Typically one should use Rmaj = 1.0 
     - **1.0**
   * - **Geometry**
     - shift
     - If igeo = 0, shift should normally be a non-negative number. It characterizes the
       Shafranov shift and is sometimes called alpha.
     - **0.0**
   * - **Geometry**
     - eps
     - This is the inverse aspect ratio of the surface in question. Used if igeo = 0.
     - **0.167**
   * - **Geometry**
     - qsf
     - This is the safety factor. Used if igeo = 0.
     - **1.4**
   * - **Geometry**
     - shat
     - This is the global magnetic shear. Used if igeo = 0.
     - **0.8**
   * - **Geometry**
     - beta
     - This is the reference beta value. Typically it would be approximately
       half of the total beta. If beta < 0., it is ignored. Only used for electromagnetic
       calculations. Not yet implemented. 
     - **-1.0**
   * - **Geometry**
     - zero_shat
     - If shat = 0 and igeo = 0, set zero_shat = true and choose shat itself to be positive and
       smaller than 1.0e-6
     - **false**
       
The **Wspectra** determines controls writes of various slices of the free energy.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Wspectra**
     - species
     - W as a function of species
     - **false**
   * - **Wspectra**
     - kx
     - W as a function of kx
     - **false**
   * - **Wspectra**
     - ky
     - W as a function of ky
     - **false**
   * - **Wspectra**
     - kz
     - W as a function of kz
     - **false**
   * - **Wspectra**
     - z
     - W as a function of z
     - **false**
   * - **Wspectra**
     - laguerre
     - W as a function of the Laguerre index
     - **false**
   * - **Wspectra**
     - hermite
     - W as a function of the Hermite index
     - **false**
   * - **Wspectra**
     - hermite_laguerre
     - W as a function of both Hermite and Laguerre indices
     - **false**
   * - **Wspectra**
     - kperp
     - W as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - **Wspectra**
     - kxky
     - W as a function of the magnitude of kx and ky.
     - **false**

The **Pspectra** determines controls writes of various slices of (1-Gamma_0) Phi**2

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Pspectra**
     - species
     - P as a function of species
     - **false**
   * - **Pspectra**
     - kx
     - P as a function of kx
     - **false**
   * - **Pspectra**
     - ky
     - P as a function of ky
     - **false**
   * - **Pspectra**
     - kz
     - P as a function of kz
     - **false**
   * - **Pspectra**
     - z
     - P as a function of z
     - **false**
   * - **Pspectra**
     - kperp
     - P as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - **Pspectra**
     - kxky
     - P as a function of the magnitude of kx and ky.
     - **false**

The **Reservoir** group controls the reservoir computing toolset:
       
.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Reservoir**
     - Use_reservoir
     - If true, train a predictor for the dynamics. Presently only set up for the Kuramoto-Sivashinsky equation.
     - **false**
   * - **Reservoir**
     - Q
     - For each real element of the quantity to be predicted, use Q reservoir elements. 
     - **20** 
   * - **Reservoir**
     - K
     - The number of elements for each row of A. 
     - **3** 
   * - **Reservoir**
     - training_steps
     - Sets the number of training steps to use. If zero, defaults to nstep/nwrite.
     - **0**
   * - **Reservoir**
     - prediction_steps
     - Sets the number of prediction timesteps.
     - **200**
   * - **Reservoir**
     - training_delta
     - Sets the reservoir timestep. If training_delta = zero, defaults to nwrite
     - **0**
   * - **Reservoir**
     - spectral_radius
     - Spectral radius of A
     - **0.6**
   * - **Reservoir**
     - regularization
     - beta parameter in the Tikhonov regularization used to calculated the weights for the output layer
     - **1.0e-4**
   * - **Reservoir**
     - input_sigma
     - Each value of the signal is multiplied by input_sigma.
       Useful for getting more dynamic range out of the tanh function. 
     - **0.5**
   * - **Reservoir**
     - noise
     - Amplitude of random noise added to the signal in the training phase.
       Default value is negative, which means no noise will be added. 
     - **-1.0**
   * - **Reservoir**
     - fake_data
     - Train on manufactured data, such as a traveling wave. 
     - **false**
   * - **Reservoir**
     - write
     - If true, write out the reservoir data, including the weights in the output layer,
       the current values in the hidden layer, the matrix A, and the input layer. Not yet implemented.
     - **false**
       
       
       
Auxiliary files
---------------
