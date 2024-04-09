Preparing a GCP VM 
++++++++++++++++++

In this tutorial we prepare a virtual machine on the Google Cloud Platform (GCP) ready to run GX.  It is assumed that the reader already has an account on GCP and is familiar with creating a virtual machine instance.

.. contents::

Creating the Virtual Machine 
----------------------------

This tutorial, the provided ``Makefiles/Makefile.gcpa2`` and the benchmarks in ``benchmarks/`` have been tested with GCP virtual machines created from scratch with the following specifications:    

* **Machine configuration:**
 
  * **Machine type (single GPU):** a2-highgpu-1g: 12 vCPU, 6 core, 85 GB memory 
  * **Machine type (dual GPU):** a2-highgpu-2g: 24 vCPU, 12 core, 170 GB memory
  * **Architecture:** x86
  * **GPU type:** NVIDIA A100 40GB
  * **Display device:** none 

*  **Boot disk:**
  * **Operating system:** Ubuntu
  * **Version:** Ubuntu 22.04 LTS, x86/64, amd64 jammy image
  * **Boot disk type:** Standard persitent disk or faster
  * **Size:** At least 50 GB

Although not tested yet, this tutorial could probably work on a different virtual machine configurations, including on H100-based machine, with minimal tweaks.
 

Installing Packages 
-------------------

NVIDIA HPC SDK
==============

We are using version 23.11 for this tutorial.  The makefile and the suggested modification to various configuation files in this tutorial are highly dependant on the selected version. See `<https://developer.nvidia.com/hpc-sdk-downloads>`_ for additional details. Run the following commands one by one in your shell:: 

     $ curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o  /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
     $ echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
     $ sudo apt-get update -y
     $ sudo apt-get install -y nvhpc-23-11

NVIDIA Drivers
==============

We are using the recommended drivers. Run the following commands one by one in your shell:: 

     $ sudo apt install -y make
     $ sudo apt install -y ubuntu-drivers-common
     $ sudo ubuntu-drivers autoinstall

Verify that drivers are correctly installed.  You should see a table with the detected GPU configuration with the following command::

     $ sudo nvidia-smi

If you have multiple GPUs, you can verify that their interconnection is correctly usign NVLink. You should see NVxy bettween GPUs by running the following command::

     $ sudo nvidia-smi topo -m

HDF5
====

Using the parallel version.  Run the following command::

     $ sudo apt install -y libhdf5-mpi-dev

NetCDF-4
========

Using the parallel version.  Run the following command::

     $ sudo apt install -y libnetcdf-mpi-dev

GSL
===

Run the following command::

     $ sudo apt install -y libgsl-dev

MPI
===

The NVIDIA HPC SDK already includes a version of OpenMPI.  So no need to install an extra one.


Python Environnment
===================

We will use Anaconda to prepare a Python environment suitable for GX. 

Verify the latest version of Anaconda for Ubuntu x86 at `<https://repo.anaconda.com/archive/>`_.  Assuming ``aconda3-2024.02-1-Linux-x86_64.sh``, from your home directory ``~/`` run the following command::

     $  wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

And then install the package with::

     $ bash Anaconda3-2024.02-1-Linux-x86_64.sh

Follow instruction to accept the license.  Use the default install directory.  Answer ``yes`` when suggested to modify your profile (last question).

Prepare the Python environnement with the commands::

     $ source .bashrc
     $ conda create -n gxenv python=3.11 numpy matplotlib scipy netCDF4 sphinx sphinx_rtd_theme

And then activate the environment with::

     $ conda activate gxenv

Final Configuration
-------------------

Add the following lines to you ``.bashrc`` file.

.. code-block:: bash

  #For GX and NVIDIA HPC SDK
  export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/math_libs/lib64
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/nccl/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/lib64
  PATH=$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/mpi/bin
  export GK_SYSTEM='gcpa2'
  conda activate gxenv

If not already existing, create the following configuration file ``~/.config/matplotlib/matplotlibrc``. Add the lines

.. code-block:: bash

  #backend as default one does not seems to load 
  backend : TkAgg

Completing Setup
----------------

Reboot the virtual machine with::

     $ sudo reboot

Once rebooted, you can reconnect. 

Compiling and Running GX
------------------------

Get GX source code with::

     $ git clone https://bitbucket.org/gyrokinetics/gx

From within ``gx`` directory, build with::

     $ make

To run gx on dual GPU, you will have to use the command::

     $ mpiexec -n 2 path\gx


