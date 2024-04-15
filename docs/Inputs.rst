Inputs
======

.. _input_file:

The GX input file
------------------

The main GX input file is parsed with `toml <https://github.com/ToruNiina/toml11>`_. Input file names should be suffixed with ``.in``, such as ``example.in``. Such an input file can be run via

.. code-block:: bash

  [/path/to/]gx example.in

This will generate an output file in NetCDF format called ``example.nc``.

The toml standard allows one to group related input variables. There are multiple
ways to specify such groupings. Since there are sample input files provided
with GX, here we will pick one particular syntactical pattern. Users are free
to use anything that is compatible with the `toml <https://github.com/ToruNiina/toml11>`_ standard.

Example input files can be found in the :ref:`quickstart` pages, and in the ``benchmarks`` `directory <https://bitbucket.org/gyrokinetics/gx/src/gx/benchmarks/>`_ in the GX repository.

A typical input file will be of the form

.. code-block:: toml

  debug = false
  
  [Dimensions]
  nx = ...
  ...

  [Domain]
  ...

  ...

Most of the parameters belong to groups, e.g. ``[Dimensions]`` or ``[Domain]``. 

In the following we describe each possible input parameter in each group. In practice, many of these parameters can be left unspecified so that default values are used.

``debug`` is a special parameter that does not belong to a group, and must be specified at the top of the input file before any groups:

.. list-table::
   :widths: 20, 20, 50, 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - *none*
     - ``debug``
     - If true, print debug information to standard out
     - **false**

Dimensions
++++++++++

The ``[Dimensions]`` group controls the number of grid-points/spectral basis functions in each dimension, and the number of evolved kinetic species.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Dimensions]``
     - ``nx``
     - The number of real-space grid points in the *x* direction. Related to the number of de-aliased Fourier modes ``nkx`` via ``nkx = 1 + 2*(nx-1)/3``. Recommended for nonlinear calculations (instead of ``nkx``).
     - 
   * - ``[Dimensions]``
     - ``ny``
     - The number of real-space grid points in the *y* direction. Related to the number of de-aliased Fourier modes ``nky`` via ``nky =  1 + (ny-1)/3``. Recommended for nonlinear calculations (instead of ``nky``).
     - 
   * - ``[Dimensions]``
     - ``nkx``
     - The number of (de-aliased) Fourier modes in the *x* direction. Recommended for linear calculations (instead of ``nx``).
     - 
   * - ``[Dimensions]``
     - ``nky``
     - The number of (de-aliased) Fourier modes in the *y* direction. Recommended for linear calculations (instead of ``ny``).
     - 
   * - ``[Dimensions]``
     - ``ntheta``
     - The number of grid points used in the *z* direction.
     - **32**
   * - ``[Dimensions]``
     - ``nhermite``
     - The number of Hermite basis functions used (:math:`v_\parallel` resolution)
     - **4**
   * - ``[Dimensions]``
     - ``nlaguerre``
     - The number of Laguerre basis functions used (:math:`\mu B` resolution)
     - **2**
   * - ``[Dimensions]``
     - ``nspecies``
     - The number of kinetic species used.
     - **1**
   * - ``[Dimensions]``
     - ``nperiod``
     - The number of poloidal turns used. ``nperiod=1`` recommended for nonlinear calculations.
     - **1**

Domain
++++++

The ``[Domain]`` group controls the physical extents of the simulation domain and the boundary conditions.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Domain]``
     - ``y0``
     - Controls box length in the binormal coordinate :math:`y` via :math:`L_y = 2\pi \texttt{y0}` (in units of :math:`\rho_\mathrm{ref}`). Also controls the minimum :math:`k_y`, so that :math:`k_{y\,\mathrm{min}}\rho_\mathrm{ref} = 1/\texttt{y0}`. 
     - **10.0**
   * - ``[Domain]``
     - ``jtwist``
     - The twist-and-shift boundary condition is controlled by ``jtwist``. This also effectively sets ``x0`` via :math:`\texttt{x0} = \texttt{y0}\,\texttt{jtwist}/(2\pi\texttt{shat})`. Typically,
       if the magnetic shear :math:`\hat{s}` is finite, you should set ``jtwist`` to be an integer close to :math:`2 \pi \hat{s}`, which will make :math:`\texttt{x0}\approx \texttt{y0}` (the default behavior).
     - If finite magnetic shear: :math:`\texttt{round}(2\pi\texttt{shat})`
        
       If zero magnetic shear: :math:`2\, \texttt{nx}`
   * - ``[Domain]``
     - ``x0``
     - Controls box length in the radial coordinate :math:`x` via :math:`L_x = 2\pi \texttt{x0}` (in units of :math:`\rho_\mathrm{ref}`). Also controls the minimum :math:`k_x`, so that :math:`k_{x\,\mathrm{min}}\rho_\mathrm{ref} = 1/\texttt{x0}`. Typically ``jtwist`` should be set instead of ``x0`` if the magnetic shear is finite.
     - If finite magnetic shear: :math:`\texttt{y0}\,\texttt{jtwist}/(2\pi\texttt{shat})`

       If zero magnetic shear: :math:`\texttt{y0}`
   * - ``[Domain]``
     - ``boundary``
     - | Controls the boundary condition in the :math:`z` (parallel) direction. The following options are possible: 
       | - ``"linked"``: twist-and-shift BC, with generalization for non-axisymmetric geometry (Martin et al 2018).
       | - ``"forced periodic"`` (or simply ``"periodic"``): use periodic BCs with no cutting of flux tube. 
       | - ``"exact periodic"``: cut flux tube at a location where gds21 = 0, and then use periodic BCs (currently for VMEC geometry only). 
       | - ``"continuous drifts"``: cut flux tube at a location where gbdrift0 = 0, and then use (generalized) twist-and-shift BC (currently for VMEC geometry only). 
       | - ``"fix aspect"``: cut flux tube at a location where y0/x0 takes the desired value, and then use (generalized) twist-and-shift BC (VMEC geometry only).
     - **"linked"**
   * - ``[Domain]``
     - ``zp``
     - Number of :math:`2\pi` segments in the :math:`z` (parallel) direction. Usually ``nperiod`` should be set instead.
     - :math:`2\,\texttt{nperiod}-1`

Physics
+++++++

Parameters that control physics options are specified in the ``[Physics]`` group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Physics]``
     - ``beta``
     - This is the reference beta value, :math:`\beta_\mathrm{ref} = 8\pi n_\mathrm{ref} T_\mathrm{ref}/B^2`. Typically it would be approximately
       half of the total beta. If ``beta > 0.0`` then electromagnetic terms will be used (depending on ``fapar`` and ``fbpar`` values); otherwise, if ``beta <= 0.``, it is ignored and the calculation is electrostatic.
     - **0.0**
   * - ``[Physics]``
     - ``nonlinear_mode``
     - Set to true to include nonlinear terms in the equations. 
     - **false**
   * - ``[Physics]``
     - ``fphi``
     - Factor multiplying electrostatic potential :math:`\Phi`
     - **1.0**
   * - ``[Physics]``
     - ``fapar``
     - Factor multiplying electromagnetic vector potential :math:`A_\parallel`
     - **0.0**
   * - ``[Physics]``
     - ``fbpar``
     - Factor multiplying electromagnetic compressional fluctuation :math:`\delta B_\parallel`
     - **0.0**

  
Time
+++++

Parameters that control the time-stepping are set in the ``[Time]`` group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Time]``
     - ``dt``
     - The maximum timestep allowed. The actual timestep will be set by stability constraints.
     - **0.05**
   * - ``[Time]``
     - ``nstep``
     - The number of timesteps to take. Set this or ``t_max`` to control the length of the simulation.
     - 
   * - ``[Time]``
     - ``t_max``
     - End time of the simulation, in code units. Set this or ``nstep`` to control the length of the simulation.
     - 
   * - ``[Time]``
     - ``t_add``
     - If restarting the simulation, run for an additional ``t_add`` time units.
     - 
   * - ``[Time]``
     - ``scheme``
     - | This string variable chooses the time-stepping scheme to be used. Choices are:
       | ``"rk3"``: 3rd order Runge-Kutta (recommended)
       | ``"rk4"``: 4th order Runge-Kutta
       | ``"sspx3"``: 3rd order SSP Runge-Kutta with extended stability 
       | ``"k10"``: Ketcheson's 10 stage, 4th order low-storage method. Useful for problems with memory constraints.
     - **"rk3"**
   * - ``[Time]``
     - ``stages``
     - The number of Runge-Kutta stages to be used for certain time-stepping schemes. Not relevant
       for most choices of ``scheme``. 
     - **10**    
   * - ``[Time]``
     - ``cfl``
     - Safety cushion factor on timestep size. The timestep used will be the computed allowable timestep multiplied by ``cfl``. If encountering numerical issues, decreasing ``cfl`` is one of the first things to try!
     - **0.9**    

Initialization
+++++++++++++++

The ``[Initialization]`` group controls the initial conditions. 

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Initialization]``
     - ``init_field``
     - The initial perturbation is applied to this component of the distribution function.
     - **"density"**    
   * - ``[Initialization]``
     - ``init_amp``
     - The amplitude of the initial perturbation.
     - **1.0e-5**    
   * - ``[Initialization]``
     - ``ikpar_init``
     - Parallel wavenumber of the initial perturbation
     - **0**
   * - ``[Initialization]``
     - ``init_electrons_only``
     - Only apply initial perturbations to electrons (when using a kinetic electron species)
     - **false**
   * - ``[Initialization]``
     - ``random_init``
     - Use completely random initial conditions (e.g. no mode structure in :math:`z`)
     - **false**

Geometry
++++++++

The ``[Geometry]`` group controls the simulation geometry. 

.. For more details about geometry options, see :ref:`geo`.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``geo_option``
     - String specifying the geometry setup. Options are ``{"s-alpha", "miller", "vmec", "eik", "nc", "slab", "const-curv", "gs2_geo"}``
     - **"miller"**    

Each ``geo_option`` has some parameters that control the geometry specification. The following groups of parameters are the parameters that can/should be specified for a particular choice of ``geo_option``. All parameters live in the ``[Geometry]`` block. Note that some parameters overlap between different ``geo_option``'s.

geo_option = "miller"
~~~~~~~~~~~~~~~~~~~~~~~~~~

This option specifies a local Miller equilibrium for a tokamak.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``Rmaj``
     - The ratio of the major radius at the center of the flux surface to the equilibrium-scale reference length, :math:`R/L_\mathrm{ref}`. 
       The choice of ``Rmaj`` can effectively choose the definition of :math:`L_\mathrm{ref}`.
       Setting ``Rmaj = 1.0`` effectively sets :math:`L_\mathrm{ref} = R`. 
       Setting ``Rmaj`` to the aspect ratio :math:`R/a` (with :math:`a` the minor radius) effectively sets :math:`L_\mathrm{ref} = a`.
     - **1.0**
   * - ``[Geometry]``
     - ``qinp``
     - The magnetic safety factor, :math:`q = (r/R)(B_t/B_p)`.
     - **1.4**
   * - ``[Geometry]``
     - ``shat``
     - The global magnetic shear, :math:`\hat{s} = (r/q) dq/dr`. 
     - **0.8**
   * - ``[Geometry]``
     - ``shift``
     - This characterizes the
       Shafranov shift and is sometimes called alpha. It should normally be a non-negative number.
     - **0.0**
   * - ``[Geometry]``
     - ``rhoc``
     - Flux surface label given by ratio of midplane diameter to the reference length, :math:`\rho = r/L_\mathrm{ref}`. 
     - 
   * - ``[Geometry]``
     - ``R_geo``
     - Major radius of magnetic field reference point, normalized to :math:`L_\mathrm{ref}` (i.e. :math:`B_t(R_\mathrm{geo}) = B_\mathrm{ref}`). 
     - 
   * - ``[Geometry]``
     - ``akappa``
     - Elongation of flux surface. 
     - 
   * - ``[Geometry]``
     - ``akappri``
     - Radial gradient of elongation of flux surface.
     - 
   * - ``[Geometry]``
     - ``tri``
     - Triangularity of flux surface.
     - 
   * - ``[Geometry]``
     - ``tripri``
     - Radial gradient of triangularity of flux surface.
     - 
   * - ``[Geometry]``
     - ``betaprim``
     - Radial gradient of equilibrium pressure. Used in the calculation of the Shafranov shift. 
     - 

geo_option = "vmec"
~~~~~~~~~~~~~~~~~~~~~~~~~~

This option generates the geometry from a VMEC equilibrium output file in NetCDF format. This is typically used for stellarators. See also the ``boundary`` parameter in ``[Domain]``, where some options are only available for VMEC geometry.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``vmec_file``
     - (Path to) the VMEC equilibrium NetCDF output file.
     - 
   * - ``[Geometry]``
     - ``alpha``
     - Field line label. 
     - **0.0**
   * - ``[Geometry]``
     - ``npol``
     - Number of poloidal turns. May be reduced based on ``boundary``.
     - **1.0**
   * - ``[Geometry]``
     - ``desired_normalized_toroidal_flux`` or ``torflux``
     - Flux surface label given by :math:`\rho = \sqrt{\Psi/\Psi_{LCFS}}`, where :math:`\Psi` is the toroidal flux.
     - **0.25**
   * - ``[Geometry]``
     - ``vmec_surface_option``
     - Controls how the flux surface is chosen. 0 = use exact radius requested. 1 = use nearest value of the VMEC half grid. 2 = use nearest value of the VMEC full grid.
     - **0**
   * - ``[Geometry]``
     - ``which_crossing``
     - For ``boundary = "exact periodic"``, ``boundary = "continuous drifts"``, or ``boundary = "fix aspect"``, this controls which crossing of the relevant quantity to cut the flux tube at. Negative values count from the end of the flux tube.
     - **-1**
   * - ``[Geometry]``
     - ``verbose``
     - Set to 1 to print lots of information from VMEC geometry module.
     - **0**

geo_option = "eik"
~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the geometry arrays from a text file in GS2's eik.out format.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``geo_file``
     - (Path to) eik geometry file.
     - "eik.out"


geo_option = "nc"
~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the geometry arrays from a NetCDF file.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``geo_file``
     - (Path to) NetCDF geometry file.
     - 
     

geo_option = "s-alpha", geo_option = "const-curv"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``eps``
     - Flux surface label given by the inverse aspect ratio of the surface in question, :math:`\epsilon = r/R`.
     - **0.167**
   * - ``[Geometry]``
     - ``Rmaj``
     - The ratio of the major radius at the center of the flux surface to the equilibrium-scale reference length, :math:`R/L_\mathrm{ref}`.  The choice of ``Rmaj`` can effectively choose the definition of :math:`L_\mathrm{ref}`. Setting ``Rmaj = 1.0`` effectively sets :math:`L_\mathrm{ref} = R`. Setting ``Rmaj`` to the aspect ratio :math:`R/a` (with :math:`a` the minor radius) effectively sets :math:`L_\mathrm{ref} = a`.
     - **1.0**
   * - ``[Geometry]``
     - ``qinp``
     - The magnetic safety factor, :math:`q = (r/R)(B_t/B_p)`.
     - **1.4**
   * - ``[Geometry]``
     - ``shat``
     - The global magnetic shear, :math:`\hat{s} = (r/q) dq/dr`. 
     - **0.8**
   * - ``[Geometry]``
     - ``shift``
     - This characterizes the Shafranov shift and is sometimes called alpha. It should normally be a non-negative number.
     - **0.0**


geo_option = "gs2_geo"
~~~~~~~~~~~~~~~~~~~~~~~~~~

This option calls GS2's eiktest executable to run GS2's geometry module directly. This is useful for generating geometry from a g-eqdsk equilibrium file, which is currently not natively supported in GX. This option requires separately building GS2's eiktest and setting ``GS2_PATH`` in the Makefile. For parameter definitions, see the GS2 documentation, or see the comments at the top of src/geo/geometry.f90 in the GS2 source code.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Geometry]``
     - ``rhoc``
     - Value of flux surface label :math:`\rho`, depending on choice of ``irho``
     - **0.5**    
   * - ``[Geometry]``
     - ``irho``
     - | Choose flux surface label:
       | case (1) :: sqrt(toroidal flux)/sqrt(toroidal flux of LCFS)
       | case (2) :: diameter/(diameter of LCFS).  recommended
       | case (3) :: poloidal flux/(poloidal flux of LCFS)
     - **2**    
   * - ``[Geometry]``
     - ``eqfile``
     - (Path to) numerical equilibrium file to read
     -     
   * - ``[Geometry]``
     - ``iflux``
     - Use ``iflux=1`` to use a numerical equilibrium (EFIT, CHEASE, etc)
     - **0**    
   * - ``[Geometry]``
     - ``geoType``
     - 
     - **0**    
   * - ``[Geometry]``
     - ``delrho``
     - 
     - **0.01**    
   * - ``[Geometry]``
     - ``bishop``
     - 
     - **0**    
   * - ``[Geometry]``
     - ``isym``
     - 
     - **0**    
   * - ``[Geometry]``
     - ``s_hat_input``
     - 
     - **1.0**    
   * - ``[Geometry]``
     - ``p_prime_input``
     - 
     - **-2.0**    
   * - ``[Geometry]``
     - ``invLp_input``
     - 
     - **5.0**    
   * - ``[Geometry]``
     - ``alpha_input``
     - 
     - **0.0**    
   * - ``[Geometry]``
     - ``efit_eq``
     - Set true to use an EFIT (e.g. g-eqdsk) equilibrium.
     - **false**    
   * - ``[Geometry]``
     - ``dfit_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``gen_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``ppl_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``local_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``idfit_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``chs_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``transp_eq``
     - 
     - **false**    
   * - ``[Geometry]``
     - ``gs2d_eq``
     - 
     - **false**    

Species
+++++++

The ``[species]`` group specifies parameters like charge, mass, and gradients of each kinetic species. For each parameter, an array is provided with entries for each species. Note that only the first ``nspecies`` elements will be read. For example, a typical ``[species]`` group might look like

.. code-block:: toml

  # it is okay to have extra species data here; only the first nspecies elements of each item are used
  [species]
   z     = [ 1.0,      -1.0     ]         # charge (normalized to Z_ref)
   mass  = [ 1.0,       2.7e-4  ]         # mass (normalized to m_ref)
   dens  = [ 1.0,       1.0     ]         # density (normalized to dens_ref)
   temp  = [ 1.0,       1.0     ]         # temperature (normalized to T_ref)
   tprim = [ 2.49,      2.49    ]         # temperature gradient, L_ref/L_T
   fprim = [ 0.8,       0.8     ]         # density gradient, L_ref/L_n
   vnewk = [ 0.0,       0.0     ]         # collision frequency
   type  = [ "ion",  "electron" ]         # species type

If ``nspecies=1``, only the first values of each table will be used (to set up the ion species). If ``nspecies=2`` the electron data will be used as well.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[species]``
     - ``z``
     - The charge (normalized to :math:`Z_\mathrm{ref}`)
     -
   * - ``[species]``
     - ``mass``
     - The mass (normalized to :math:`m_\mathrm{ref}`)
     -
   * - ``[species]``
     - ``dens``
     - The density (normalized to :math:`n_\mathrm{ref}`)
     -
   * - ``[species]``
     - ``temp``
     - The temperature (normalized to :math:`T_\mathrm{ref}`)
     - **1.0**
   * - ``[species]``
     - ``tprim``
     - Temperature gradient, :math:`L_\mathrm{ref}/L_T`
     -
   * - ``[species]``
     - ``fprim``
     - Density gradient, :math:`L_\mathrm{ref}/L_n`
     - 
   * - ``[species]``
     - ``uprim``
     - Velocity gradient, :math:`L_\mathrm{ref}/L_u`. **Not yet implemented**.
     - **0.0**
   * - ``[species]``
     - ``vnewk``
     - The collision frequency
     - **0.0**
   * - ``[species]``
     - ``type``
     - The type of species, such as ``"ion"`` or ``"electron"``
     - **"ion"**

Boltzmann
++++++++++

The ``[Boltzmann]`` group sets up a Boltzmann species, which can be either electrons (e.g. for ITG calculations) or ions (e.g. for ETG calculations).

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Boltzmann]``
     - ``add_Boltzmann_species``
     - If true, include a species with a Boltzmann response
     - **false**
   * - ``[Boltzmann]``
     - ``Boltzmann_type``
     - Specify the Boltzmann species type. Choose either ``"electrons"`` or ``"ions"``.
     - **"electrons"**
   * - ``[Boltzmann]``
     - ``tau_fac``
     - Set the value of :math:`\tau = T_\mathrm{ref}/T_\mathrm{Boltzmann}` (which is usually :math:`\tau = T_\mathrm{kinetic}/T_\mathrm{Boltzmann}`).
     - **1.0**

Dissipation
+++++++++++

The ``[Dissipation]`` group controls numerical dissipation parameters, including (mostly experimental) closure models. For more details, see :ref:`diss` and :ref:`closures`.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Dissipation]``
     - ``hypercollisions``
     - If true, use hypercollision model to provide hyper-dissipation at grid-scales in velocity space. For details, see :ref:`diss`.
     - **false**
   * - ``[Dissipation]``
     - ``nu_hyper_l``
     - If ``hypercollisions=true``, sets strength of Laguerre hypercollisions at grid-scales in :math:`\mu B`. 
     - **0.5**
   * - ``[Dissipation]``
     - ``nu_hyper_m``
     - If ``hypercollisions=true``, sets strength of Hermite hypercollisions at grid-scales in :math:`v_\parallel`. 
     - **0.5**
   * - ``[Dissipation]``
     - ``p_hyper_l``
     - If ``hypercollisions=true``, sets exponent of Laguerre hypercollisions.
     - **6**
   * - ``[Dissipation]``
     - ``p_hyper_m``
     - If ``hypercollisions=true``, sets exponent of Hermite hypercollisions.
     - **6**
   * - ``[Dissipation]``
     - ``hyper``
     - If true, use a simple hyperdiffusivity model to provide hyper-dissipation at grid-scales in the perpendicular dimensions. Recommended only for nonlinear calculations. For details, see :ref:`diss`.
     - **false**
   * - ``[Dissipation]``
     - ``D_hyper``
     - If ``hyper=true``, sets strength of hyperdiffusivity
     - **0.5**
   * - ``[Dissipation]``
     - ``p_hyper``
     - If ``hyper=true``, sets exponent of hyperdiffusivity to ``2*p_hyper``.
     - **2**
   * - ``[Dissipation]``
     - ``HB_hyper``
     - If true, use the Hammett-Belli hyperdiffusivity model to provide hyper-dissipation at grid-scales in the perpendicular dimensions. Recommended only for nonlinear calculations. For details, see :ref:`diss`.
     - **false**
   * - ``[Dissipation]``
     - ``D_HB``
     - If ``HB_hyper=true``, sets the strength of the H-B hyperdiffusivity model.
     - **1.0**
   * - ``[Dissipation]``
     - ``w_osc``
     - If ``HB_hyper=true``, sets the frequency parameter in the H-B model.
     - **0.0**
   * - ``[Dissipation]``
     - ``p_HB``
     - If ``HB_hyper=true``, sets the exponent for the H-B model,
       where 2 corresponds to the fourth power of k.
     - **2**
   * - ``[Dissipation]``
     - ``closure_model``
     - Closure model to use. For options, see :ref:`closures`.
     - **"none"**
   * - ``[Dissipation]``
     - ``smith_par_q``
     - A parameter for ``closure_model = "smith_par"``. 
     - **3**
   * - ``[Dissipation]``
     - ``smith_perp_q``
     - A parameter for ``closure_model = "smith_perp"``. 
     - **3**


Restart
+++++++

The ``[Restart]`` group controls reading and writing of data for restarting (continuing) from previous runs. **Warning**: restart capability is currently limited. See this `issue <https://bitbucket.org/gyrokinetics/gx/issues/1/append-on-restart-instead-of-overwrite.>`_.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Restart]``
     - ``restart``
     - Set to true to continue the simulation from a previous run. The simulation will be initialized to the previous state by reading data from the file specified by ``restart_from_file``.
     - **false**
   * - ``[Restart]``
     - ``save_for_restart``
     - Set to true to write information needed for a future restart. File is written every ``nsave`` timesteps.
     - **true**
   * - ``[Restart]``
     - ``restart_to_file``
     - Filename to write data needed for restarting the present run.       
     - **"[input_stem].restart.nc"**
   * - ``[Restart]``
     - ``restart_from_file``
     - Filename for a file written from a previous run, to be read and used to continue (restart) the previous run.
     - **"[input_stem].restart.nc"**
   * - ``[Restart]``
     - ``nsave``
     - Restart data will be written every ``nsave`` steps.
     - max(1, :math:`\texttt{nstep}`/10)
   * - ``[Restart]``
     - ``scale``
     - Multiply all variables in the restart data by a factor of ``scale``. 
     - **1.0**

Diagnostics
+++++++++++

The ``[Diagnostics]`` group controls the diagnostic quantities that are computed and written to the NetCDF output file. For more details about diagnostic quantities and options, see :ref:`diag`.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Diagnostics]``
     - omega
     - Write instantaneous estimates of the complex frequency for each Fourier component of the electrostatic potential.
     - **false**
   * - ``[Diagnostics]``
     - free_energy
     - Write the total free energy (integrated over the phase-space domain and summed over species) as a function of time. 
     - **true**
   * - ``[Diagnostics]``
     - fluxes
     - Write the turbulent fluxes for each species. 
     - **false**
   * - ``[Diagnostics]``
     - fixed_amplitude
     - Periodically rescale amplitude of phi to avoid overflow. Only for linear calculations.
     - **false**
   * - ``[Diagnostics]``
     - all_zonal_scalars
     - Write quantities such as the RMS value of the zonal component of the ExB velocity as function of time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - ``[Diagnostics]``
     - avg_zvE
     - Write the RMS value of the zonal component of ExB velocity as a function time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkvE
     - Write the RMS value of the zonal component of the shear of the ExB velocity as a function time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkden
     - Write the RMS value of the zonal component of the guiding center
       radial density gradient as a function of time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkUpar
     - Write the RMS value of the zonal component of the guiding center
       parallel velocity as a function of time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkTpar
     - Write the RMS value of the zonal component of the guiding center
       parallel temperature as a function of time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkTperp
     - Write the RMS value of the zonal component of the guiding center
       perpendicular temperature as a function of time.
     - **false**
   * - ``[Diagnostics]``
     - avg_zkqpar
     - Write the RMS value of the zonal component of the guiding center
       parallel-parallel heat flux as a function of time.
     - **false**
   * - ``[Diagnostics]``
     - all_zonal
     - Write quantities such as the zonal component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - ``[Diagnostics]``
     - vE
     - Write the zonal component of ExB velocity as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kvE
     - Write the zonal component of the shear of the ExB velocity as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kden
     - Write the zonal component of the radial gradient of the guiding center
       density as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kUpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel velocity as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kTpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel temperature as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kTperp
     - Write the zonal component of the radial gradient of the guiding center
       perpendicular temperature as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - kqpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel-parallel heat flux as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - all_non_zonal
     - Write quantities such as the non-zonal *y*-component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - ``[Diagnostics]``
     - xyvEx
     - Write the *x*-component of the ExB velocity as a function of *x*, *y*, and time.
     - **false**
   * - ``[Diagnostics]``
     - xyvEy
     - Write the non-zonal *y*-component of the ExB velocity as a function of *x*, *y*, and time.
     - **false**
   * - ``[Diagnostics]``
     - xykvE
     - Write the non-zonal part of the shear of the *y*-component of the ExB velocity as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - xyden
     - Write the non-zonal component of the guiding center density as a function of *x* and time.       
     - **false**
   * - ``[Diagnostics]``
     - xyUpar
     - Write the non-zonal component of the guiding center parallel velocity as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - xyTpar
     - Write the non-zonal component of the guiding center parallel temperature as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - xyTperp
     - Write the non-zonal component of the guiding center perpendicular temperature as a function of *x* and time.
     - **false**
   * - ``[Diagnostics]``
     - xyqpar
     - Write the non-zonal component of the guiding center parallel-parallel heat flux as a function of *x* and time.
     - **false**

The ``[Wspectra]`` group controls writes of various slices of :math:`W_s(k_x,k_y,z, \ell, m) = |G_{\ell,m\,s}|^2`, the moment component of the free energy.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Wspectra]``
     - species
     - W as a function of species
     - **false**
   * - ``[Wspectra]``
     - kx
     - W as a function of kx
     - **false**
   * - ``[Wspectra]``
     - ky
     - W as a function of ky
     - **false**
   * - ``[Wspectra]``
     - kz
     - W as a function of kz
     - **false**
   * - ``[Wspectra]``
     - z
     - W as a function of z
     - **false**
   * - ``[Wspectra]``
     - laguerre
     - W as a function of the Laguerre index
     - **false**
   * - ``[Wspectra]``
     - hermite
     - W as a function of the Hermite index
     - **false**
   * - ``[Wspectra]``
     - hermite_laguerre
     - W as a function of both Hermite and Laguerre indices
     - **false**
   * - ``[Wspectra]``
     - kperp
     - W as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - ``[Wspectra]``
     - kxky
     - W as a function of the magnitude of kx and ky.
     - **false**

The ``[Pspectra]`` group controls writes of various slices of :math:`P_s(k_x,k_y,z) = [1-\Gamma_0(b_s)] |\Phi|^2`, the field component of the free energy.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Pspectra]``
     - species
     - P as a function of species
     - **false**
   * - ``[Pspectra]``
     - kx
     - P as a function of kx
     - **false**
   * - ``[Pspectra]``
     - ky
     - P as a function of ky
     - **false**
   * - ``[Pspectra]``
     - kz
     - P as a function of kz
     - **false**
   * - ``[Pspectra]``
     - z
     - P as a function of z
     - **false**
   * - ``[Pspectra]``
     - kperp
     - P as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - ``[Pspectra]``
     - kxky
     - P as a function of the magnitude of kx and ky.
     - **false**

The ``[Qspectra]`` group controls writes of various slices of :math:`Q_s(k_x,k_y,z)`, the heat flux.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Qspectra]``
     - species
     - Q as a function of species
     - **false**
   * - ``[Qspectra]``
     - kx
     - Q as a function of kx
     - **false**
   * - ``[Qspectra]``
     - ky
     - Q as a function of ky
     - **false**
   * - ``[Qspectra]``
     - kz
     - Q as a function of kz
     - **false**
   * - ``[Qspectra]``
     - z
     - Q as a function of z
     - **false**
   * - ``[Qspectra]``
     - kperp
     - Q as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - ``[Qspectra]``
     - kxky
     - Q as a function of the magnitude of kx and ky.
     - **false**

The ``[Gamspectra]`` group controls writes of various slices of :math:`\Gamma_s(k_x,k_y,z)`, the particle flux.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Gamspectra]``
     - species
     - :math:`\Gamma` as a function of species
     - **false**
   * - ``[Gamspectra]``
     - kx
     - :math:`\Gamma` as a function of kx
     - **false**
   * - ``[Gamspectra]``
     - ky
     - :math:`\Gamma` as a function of ky
     - **false**
   * - ``[Gamspectra]``
     - kz
     - :math:`\Gamma` as a function of kz
     - **false**
   * - ``[Gamspectra]``
     - z
     - :math:`\Gamma` as a function of z
     - **false**
   * - ``[Gamspectra]``
     - kperp
     - :math:`\Gamma` as a function of the magnitude of kperp. Not yet implemented.
     - **false**
   * - ``[Gamspectra]``
     - kxky
     - :math:`\Gamma` as a function of the magnitude of kx and ky.
     - **false**

Expert
+++++++

The ``[Expert]`` group controls parameters reserved for expert users.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Expert]``
     - i_share
     - An integer related to the shared memory block used for the inner loop of the linear solver.
     - **8**
   * - ``[Expert]``
     - nreal 
     - Enforce the reality condition every ``nreal`` timesteps. 
     - **1**
   * - ``[Expert]``
     - dealias_kz
     - If true, dealias in :math:`k_z`. Only available for ``boundary="periodic"``.
     - **false**
   * - ``[Expert]``
     - local_limit
     - If true, run calculation in local limit, where :math:`k_z` is a scalar parameter given by :math:`k_z = 1.0/\texttt{zp}`.
     - **false**
   * - ``[Expert]``
     - init_single
     - Only initialize a single Fourier mode if true
     - **false**
   * - ``[Expert]``
     - ikx_single
     - Index of the kx mode to be initialized if ``init_single`` is true
     - **0**
   * - ``[Expert]``
     - iky_single
     - Index of the ky mode to be initialized if ``init_single`` is true
     - **1**
   * - ``[Expert]``
     - eqfix
     - Do not evolve some particular Fourier harmonic
     - **false**
   * - ``[Expert]``
     - ikx_fixed
     - Index of the kx mode to be fixed in time if ``eqfix`` is true
     - **-1**
   * - ``[Expert]``
     - iky_fixed
     - Index of the ky mode to be fixed in time if ``eqfix`` is true
     - **-1**
   * - ``[Expert]``
     - phi_ext
     - Value of phi to use for a Rosenbluth-Hinton test
     - **0.0**
   * - ``[Expert]``
     - source
     - Used to specify various kinds of tests
     - **"default"**
   * - ``[Expert]``
     - t0
     - The time at which to start changing ``tprim``
     - **-1.0**
   * - ``[Expert]``
     - tf
     - The time at which to stop changing ``tprim``
     - **-1.0**
   * - ``[Expert]``
     - tprim0
     - The value of ``tprim`` to start with (at time ``t0``)
     - **-1.0**
   * - ``[Expert]``
     - tprimf
     - The value of ``tprim`` to end with (at time ``tf``)
     - **-1.0**

Forcing
+++++++

Add forcing with the ``[Forcing]`` group. Not generally implemented. 

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[Forcing]``
     - forcing_type
     - Picks among the forcing options
     - **"Kz"**
   * - ``[Forcing]``
     - stir_field
     - Determines which moment of the GK equation is forced
     - **"density"**
   * - ``[Forcing]``
     - forcing_amp
     - Amplitude of the forcing
     - **1.0**
   * - ``[Forcing]``
     - forcing_index
     - Index of the forcing
     - **1**
   * - ``[Forcing]``
     - no_fields
     - Turn off the field terms in the GK equation if this is true
     - **false**
