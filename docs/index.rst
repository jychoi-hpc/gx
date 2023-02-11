.. GX documentation master file, created by
   sphinx-quickstart on Wed Apr 21 21:02:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The GX code: documentation home
===============================
	     
GX is a code for solving the nonlinear gyrokinetic system for low-frequency turbulence in magnetized plasmas using Fourier-Hermite-Laguerre spectral methods.
A unique feature of GX is the use of a Hermite-Laguerre velocity discretization, which allows GX to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

Another unique feature of GX is that it is a GPU-native code, designed and optimized in CUDA/C++. 
This means you will need access to an NVIDIA GPU to run GX. 

The GX repository is open source and hosted on BitBucket: https://bitbucket.org/gyrokinetics/gx. The GX paper is now on arXiv! https://arxiv.org/abs/2209.06731

GX is currently under rapid development, resulting in quickly-changing functionality and capabilities.
A number of planned improvements to the code are listed in the issue tracker on BitBucket (https://bitbucket.org/gyrokinetics/gx/issues).

.. toctree::
   :maxdepth: 2

   Install
   Quickstart
   Reference
   Citing
   License
