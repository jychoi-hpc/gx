# VMEC to GX Geometry Interface

The purpose of this module is to calculate the required geometric quantities for GX from a VMEC equilibrium file at each grid point in the parallel coordinates. The parallel coordinate is altered so that gradpar = hat{b}\cdot\nabla_{\parallel} = const. in order to allow for FFTs along a field line. The output of this interface is a grid file that can be used as the value for the input parameter "geofilename".

gradpar = <b>&#183;&nabla;<sub>&#x2225;<\sub> = const
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

## Running the module

The only command line input is the VMEC equilibrium file, which must be in the *.nc format
```
./convert_VMEC_to_GX [vmec_file.nc]
```