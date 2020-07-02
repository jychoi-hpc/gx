#include "cuda_constants.h"

#ifdef __CUDA_ARCH__
// This set of global DEVICE constants are set when compiled for a GPU
__constant__ int nx, ny, nyc, nz, nspecies, nm, nl, zp, ikx_fixed, iky_fixed;
__constant__ float dx, dy;

#else

// This set of global constants (without the __device__ qualifier) are used when compiling host code, and allow one to test the __host__ __device__ functions with regular CPU code
int nx, ny, nyc, nz, nspecies, nm, nl, zp, ikx_fixed, iky_fixed;
float dx, dy;

#endif
