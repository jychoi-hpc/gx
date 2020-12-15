# VMEC to GX Geometry Interface

The purpose of this module is to calculate the required geometric quantities for GX from a VMEC equilibrium file at each grid point in the parallel coordinates. The parallel coordinate is altered so that gradpar = hat{b}\cdot\nabla_{\parallel} = const. (gradpar = <b>b</b>&#183;&nabla;<sub>&#8741;</sub> = const.) in order to allow for FFTs along a field line. The output of this interface is a grid file that can be used as the value for the input parameter "geofilename".

## Prerequisites

CMake

## Installing

Assuming your platform has a working version of CMake, in the GX/geometry_module directory, do the following:

```
mkdir build
cd build
cmake ..
make
```

## Input Parameters

The input parameters are currently set in the geometric_coefficients.cu source file. Therefore, if any parameters are changed, the module will need to be recompiled.

- alpha: Magnetic field line label. alpha=0.0 would correspond to a flux tube with the center at the outboard midplane and the center of one of the symmetric field periods.
- nzgrid: The number of grid points in GX will be 2*nzgrid+1
- npol: Sets limits of the flux tube to be [-npol*&pi;, npol*&pi;]
- desired_normalized_toroidal_flux
- vmec_surface_option:
  * 0 - interpolates quantities between VMEC's half and full grid to get geometric quantities at exactly the "desired_normalized_toroidal_flux" input
  * 1 - calculates quantities on the closest surface of VMEC's half grid to "desired_normalized_toroidal_flux"
  * 2 - calculates quantities on the closest surface of VMEC's full grid to "desired_normalized_toroidal_flux"

### Choosing custom flux tube sizes

If desired, one can choose the exact endpoints of the flux tube, or have the endpoints coincide with the zeros of gds21 or gbdrift0. The input parameters to control this feature are below:
- flux_tube_cut:
  * "none" - This is the default parameter corresponding to no cutting. npol will completely control the length
  * "custom" - Choose the exact endpoints of the tube
  * "gds21" - Choose a particular zero of the gds21 array as the endpoints
  * "gbdrift0" - Choose a particular zero of the gbdrift0 array as the endpoints
- custom_length: If "custom" is chosen for flux_tube_cut, the theta values for the flux tube will be [-custom_length , custom_length]
- which_crossing: If either "gds21" or "gbdrift0" are chosen for flux_tube_cut, this integer value will specify which zero to use, since these functions will have numerous crossing. Choosing "1" will yield the first zero from the center, "2" will yield the second, and so on.

## Running the module

In whatever directory the VMEC equilibrium is located, give /path/to/convert_VMEC_to_GX, with the only command line input being the VMEC equilibrium file, which must be in the *.nc format
```
/path/to/convert_VMEC_to_GX [vmec_file.nc]
```

## Test Cases

The tests/ directory contains 3 separate VMEC equilibria, as well as the corresponding outputs from the GIST code, which similarly computes geometric quantities for GENE and GS2 from a VMEC input.
One can choose any of the 3 equilibria with some combination of input paramters and compare the results from this module to the GIST