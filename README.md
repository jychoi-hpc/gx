# GX

## Introduction

GX is a code for solving the nonlinear gyrokinetic system for low-frequency turbulence in magnetized plasmas using Fourier-Hermite-Laguerre spectral methods.
A unique feature of GX is the use of a Hermite-Laguerre velocity discretization, which allows GX to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

Another unique feature of GX is that it is a GPU-native code, designed and optimized in CUDA/C++. 
This means you will need access to an NVIDIA GPU to run GX. 

User documentation for the code is available at https://gx.rtfd.io (a work in progress).

GX is currently under rapid development, resulting in quickly-changing functionality and capabilities. 
A number of planned improvements to the code are listed in the issue tracker on BitBucket (https://bitbucket.org/gyrokinetics/gx/issues).

## Dependencies

The following external dependencies are required to build GX:

- NVIDIA HPC SDK, which includes:
	- nvcc compiler
	- CUDA Runtime
	- cuFFT
	- cuBLAS
	- cuSOLVER
	- cuTENSOR
	- cuLIBOS
	- NCCL
- NetCDF (parallel)
- HDF5 (parallel)
- MPI
- GSL

## Building the code

GX is currently supported on several systems, enabling a relatively simple build process.
To see if your system (e.g. `perlmutter`, `traverse`, or `stellar`) is supported, check to see if there is a Makefile corresponding to your system in the `Makefiles` directory.

If you are on a supported system (we'll use `perlmutter` as an example):

1. Set the `GK_SYSTEM` environment variable to the name of the system. For `perlmutter`, this can be accomplished with (assuming `bash`)
```
$ export GK_SYSTEM='perlmutter'
```
2. Load required modules for the system, if available. The necessary commands can be found in the comments at the top of the `Makefiles/Makefile.[GK_SYSTEM]` file.
Typically it is easiest to add these module load commands to your `.bashrc` file (or equivalent) so that the necessary modules (which are needed at compile-time and at run-time)
are always automatically loaded.
3. Build GX by simply executing `make` (or `make -j` to build in parallel) in the main directory:
```
$ make [-j]
```
This will create the `gx` executable in the main directory.

If you are not on a supported system, you will need to make a custom `Makefiles/Makefile.[GK_SYSTEM]`
with the paths to the necessary libraries; please use ``Makefile.generic`` as a template, following the commented instructions at the top of the file. 

## Running GX

To run GX using an input file named `run_name.in`, execute
```
$ [path/to/]gx [path/to/]run_name.in
```

Sample input files for standard benchmark cases are provided in the `benchmarks` directory.
For example, to run a linear ITG adiabatic-electron miller geometry benchmark, 
navigate into the `benchmarks/linear/ITG/` directory, and execute
```
$ ../../../gx itg_miller_adiabatic_electrons.in
```
Diagnostic output will be printed to the screen, and also to the `itg_miller_adiabatic_electrons.nc` NetCDF output file.

## Setting up a Python environment for GX

Some parts of GX (e.g. post-processing scripts and some geometry modules) require a Python installation with

- python 3 (3.11 preferred)
- numpy
- scipy
- matplotlib
- netCDF4
- tomli (if python < 3.11)

We recommend using a Conda (https://conda.io/miniconda.html) environment to install the dependencies. To create a Conda environment for GX called `gxenv`, use

```
  $ conda create -n gxenv python=3.11 numpy matplotlib scipy netCDF4
```

After creating the environment (only needed once per system), one must always have the `gxenv` environment activated in order to use the GX python scripts. Activate the environment with

```
  $ conda activate gxenv
```

## Citing GX

If you use GX in your work, please cite the following papers:

N. R. Mandell, W. D. Dorland, and M. Landreman. 2018. "Laguerre-Hermite pseudo-spectral velocity formulation of gyrokinetics". J. Plasma Phys. **84**, 905840108. https://doi.org/10.1017/S0022377818000041

N. R. Mandell, W. D. Dorland, I. Abel, R. Gaur, P. Kim, M. Martin, and T. Qian. 2024. "GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design". J. Plasma Phys. **90**, 905900402. https://doi.org/10.1017/S0022377824000631

## License

Copyright (c) 2011-2022 Noah R. Mandell, William D. Dorland, and the GX team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
