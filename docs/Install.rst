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

The following external dependencies are required to build GX:

- `NVIDIA HPC SDK <https://developer.nvidia.com/hpc-sdk>`_, which includes:
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

Building the code
-----------------

GX is currently supported on several systems, enabling a relatively simple build process.
To see if your system (e.g. ``perlmutter``, ``traverse``, or ``stellar``) is supported, check to see if there is a Makefile corresponding to your system in the ``Makefiles`` directory.

If you are on a supported system (we'll use ``perlmutter`` as an example):

#. Set the ``GK_SYSTEM`` environment variable to the name of the system. For ``perlmutter``, this can be accomplished with (assuming ``bash``)::

     $ export GK_SYSTEM='perlmutter'

#. Load required modules for the system, if available. The necessary commands can be found in the comments at the top of the ``Makefiles/Makefile.[GK_SYSTEM]`` file. Typically it is easiest to add these ``module load`` commands to your ``.bashrc`` file (or equivalent) so that the necessary modules (which are needed at compile-time and at run-time) are always automatically loaded.

#. Build GX by simply executing ``make`` (or ``make -j`` to build in parallel) in the main directory: ::

     $ make [-j]

   This will create the ``gx`` executable in the main directory.


If you are not on a supported system, you will need to make a custom ``Makefiles/Makefile.[GK_SYSTEM]``
with the paths to the necessary libraries; please use ``Makefile.generic`` as a template. 

Setting up a Python environment for GX
--------------------------------------

Some parts of GX (e.g. post-processing scripts and some geometry modules) require a Python installation with

- python 3 (3.11 preferred)
- numpy
- scipy
- matplotlib
- netCDF4
- tomli (if python < 3.11)

We recommend using a `Conda <https://conda.io/miniconda.html>`_ environment to install the dependencies. To create a Conda environment for GX called ``gxenv``, use

.. code-block:: bash

  conda create -n gxenv python=3.11 numpy matplotlib scipy netCDF4

After creating the environment (only needed once per system), one must always have the ``gxenv`` environment activated in order to use the GX python scripts. Activate the environment with

.. code-block:: bash

  conda activate gxenv

.
