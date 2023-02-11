.. _install:

Installing and building GX
++++++++++++++++++++++++++

This page contains information for obtaining the GX source code and building the code.

Obtaining the source code
-------------------------

Clone the `BitBucket <https://bitbucket.org/gyrokinetics/gx>`_ repository using::

    git clone https://bitbucket.org/gyrokinetics/gx

Navigate into the ``gx`` directory to begin.

Dependencies
============

In addition to the NVCC compiler, the following external dependencies are required to build GX:

- CUDA libraries (all of these are included with the CUDA toolkit and/or the NVIDIA HPC SDK): 

	- CUDA Runtime
	- cuFFT
	- cuBLAS
	- cuSOLVER
	- cuTENSOR
	- cuLIBOS
        - NCCL

- NetCDF
- HDF5
- GSL

Building the code
-----------------

GX is currently supported on several systems, enabling a relatively simple build process.
To see if your system (e.g. ``traverse`` or ``stellar``) is supported, check to see if there is a Makefile corresponding to your system in the ``Makefiles`` directory.

If you are on a supported system (we'll use ``traverse`` as an example):

#. Set the ``GK_SYSTEM`` environment variable to the name of the system. For ``traverse``, this can be accomplished with (assuming ``bash``)::

     $ export GK_SYSTEM='traverse'

#. Load required modules for the system, if available. The necessary commands can be found in the comments at the top of the ``Makefiles/Makefile.[GK_SYSTEM]`` file. Typically it is easiest to add these ``module load`` commands to your ``.bashrc`` file (or equivalent) so that the necessary modules (which are needed at compile-time and at run-time) are always automatically loaded.

#. Build GX by simply executing ``make`` (or ``make -j`` to build in parallel) in the main directory: ::

     $ make [-j]

   This will create the ``gx`` executable in the main directory.


If you are not on a supported system, you will need to make a custom ``Makefiles/Makefile.[GK_SYSTEM]``
with the paths to the necessary libraries; please use ``Makefile.traverse`` as a template. 
