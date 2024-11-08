.. GX documentation master file, created by
   sphinx-quickstart on Wed Apr 21 21:02:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The GX code: documentation home
===============================
	     
GX is a code for solving the nonlinear gyrokinetic system for low-frequency turbulence in magnetized plasmas using Fourier-Hermite-Laguerre spectral methods.
A unique feature of GX is the use of a Hermite-Laguerre velocity discretization, which allows GX to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

Another unique feature of GX is that it is a GPU-native code, designed and optimized in CUDA/C++. 
This means you will need access to an NVIDIA GPU to run GX. GX already runs on many of the worlds largest HPC facilities -- includeing NERSC Perlmutter (https://www.nersc.gov), ALCF Polaris (https://www.alcf.anl.gov/), and CINECA Leonardo (https://www.hpc.cineca.it/systems/hardware/leonardo/).

The underlying algorithms and structure of GX are described in two main publications: https://www.arxiv.org/abs/1708.04029 and https://arxiv.org/abs/2209.06731.

GX is open source, distributed under the MIT license, and hosted on BitBucket: https://bitbucket.org/gyrokinetics/gx.

GX is under active development. To receive announcements about GX, such as updates on new features or publications, subscribe to the gx-users mailing list by contacting *iabel AT umd DOT edu*.
To contact the GX development team, you can email *gx-developers AT listserv DOT umd DOT edu*.

The list of current known bugs can be found at https://bitbucket.org/gyrokinetics/gx/issues. Please file a bug if you find one!


.. toctree::
   :maxdepth: 2

   Install
   Quickstart
   Reference
   Citing
   License
