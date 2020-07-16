# VMEC to GX Geometry Interface

The purpose of this module is to calculate the required geometric quantities for GX from a VMEC equilibrium file at each grid point in the parallel coordinates. The parallel coordinate is altered so that gradpar = \hat{b}\cdot\nabla_{\parallel} = const. in order to allow for FFTs along a field line. The output of this interface is a grid file that can be used as the value for the input parameter "geofilename".

## Prerequisites

CMake

### Installing

Assuming your platform has a working version of CMake, in the GX/geometry_module directory, do the following:

```
mkdir build
cd build
cmake ..
make
```

### Running

