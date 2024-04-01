This test runs a nonlinear ITG turbulence calculation using a circular Miller geometry with Cyclone-base-case-like parameters.
This test uses a Boltzmann adiabatic electron response.

To run the test, simply use 
```
[/path/to/]gx cyclone_miller_adiabatic_electrons.in
```

To plot the turbulent heat flux, use
```
python ../../../post_processing/heat_flux.py cyclone_miller_adiabatic_electrons.nc
```

