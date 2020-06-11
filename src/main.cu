#include <stdlib.h>
#include <stdio.h>
#ifdef USE_MPI
#include <mpi.h>
#else
#define MPI_COMM_WORLD 1
typedef int MPI_Comm; 
#endif
#include "gx_lib.h"

int main(int argc, char* argv[])
{
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  MPI_Comm mpcom_ftn = MPI_COMM_WORLD;
        
  gx_main(argc, argv, mpcom_ftn);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  cudaDeviceReset();
}
