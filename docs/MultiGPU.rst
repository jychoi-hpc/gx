.. _quickmulti:

Using multi-GPU parallelization [NEW!]
+++++++++++++++++++++++++++++++

GX now has the capability to parallelize a calculation over multiple GPUs. 

.. contents::

Constraints
-----------

Parallelization is currently only implemented over the species and Hermite indices. The computational grid must divide evenly into the number of GPUs requested, and decomposition of species index is prioritized. If :math:`N_\mathrm{sp}` is the number of species, :math:`N_m` is the number of Hermite modes, and :math:`N_\mathrm{GPU}` is the number of GPUs to be used for the calculation, this means:

- if :math:`N_\mathrm{GPU} \leq N_\mathrm{sp}`, :math:`N_\mathrm{sp}` must be an integer multiple of :math:`N_\mathrm{GPU}`
- if :math:`N_\mathrm{GPU} > N_\mathrm{sp}`, :math:`N_\mathrm{GPU}` must be an integer multiple of :math:`N_\mathrm{sp}` AND :math:`N_m` must be an integer multiple of :math:`N_\mathrm{GPU}/N_\mathrm{sp}`.

For example, if :math:`N_\mathrm{sp} = 2` and :math:`N_m = 16`, the number of GPUs can be any of :math:`N_\mathrm{GPU} = \{1,2,4,8,16,32\}`. 

Requesting a multi-GPU job (SLURM)
----------------------------------

On systems with SLURM job management, use the ``--nodes=[N]`` flag to specify the number of nodes, and the ``--gpus-per-node=[P]`` flag to specify the number of GPUs per node, so that the total number of GPUs requested is ``N*P``. The maximum number of GPUs per node is system-specific, so check the documentation for your system, but typical configurations are 2 or 4 GPUs per node.

For example, an interactive job that requests 4 GPUs on a system with 4 GPUs/node can be requested with something like

.. code-block:: bash

  salloc --nodes=1 -—gpus-per-node=4 ...

For a batch job, the submission script should include something like

.. code-block:: bash

  ...
  #SBATCH --nodes=1
  #SBATCH --gpus-per-node=4
  ...

Running the calculation
-----------------------

The number of GPUs used by the calculation is controlled at runtime by the number of MPI processes. Suppose we have a job with 4 GPUs allocated via one of the commands above. Now we can choose to use 1, 2, or 4 GPUs for the GX calculation (assuming :math:`N_\mathrm{sp}N_m\geq4`). On SLURM systems, use srun to launch the job:

.. code-block:: bash

  srun -n [NGPU] [/path/to/]gx [inputfile].in

where ``NGPU`` can be 1, 2, or 4 here.

Performance considerations
--------------------------

Performance will often be limited by the speed of the connection between GPUs. Typically GPUs within a single node are connected with a faster interconnect (e.g. NVLINK) than across nodes, so scaling efficiency may degrade somewhat when parallelizing across multiple nodes. For details about the scaling of the code, see Section 7 of the GX paper (Mandell et al., 2022). 

