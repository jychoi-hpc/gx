#include "grids.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "device_funcs.h"

/*
The moments are defined as functions of (ky, kx, z) natively.
The moments are real functions of (y, x, z). But we are working 
with Fourier components. g is real space, G is Fourier harmonic

g(x, y, z) = sum_(kx, ky) G(ky, kx, z) exp(i (kx x + ky y)) 

where G is now complex -- but the way reality is enforced is important. 

We could say 

g(x, y, z) = sum_(kx, ky) G(ky, kx, z) exp(i (kx x + ky y)) + c.c.

and let the sum run over non-negative values of kx, ky. But that is not the 
cuFFT convention. Instead, reality is enforced by setting 

G(kx, ky, z) = conjg[ G(-kx, -ky, z)                  (R)

because :

g(x, y, z) = sum_(kx = -Kx:Kx, ky = 0:Ky) G(ky, kx, z) exp(i (kx x + ky y))

Using (R):  (we will come back to ky=0 in a moment)

g(x, y, z) = sum_(kx = -Kx:Kx, ky = 0:Ky) G(ky, kx, z) exp(i (kx x + ky y))
           + sum_(kx = -Kx:Kx, ky = -Ky:0) G(ky, kx, z) exp(i (kx x + ky y))

           = sum_(kx = -Kx:Kx, ky = 0:Ky) G(ky, kx, z) exp(i (kx x + ky y))
           + sum_(kx = -Kx:Kx, ky = 0:Ky) G(-ky, kx, z) exp(i (kx x + -ky y))


           = sum_(kx = -Kx:Kx, ky = 0:Ky) G(ky, kx, z) exp(i (kx x + ky y))
                                        + G*(ky, kx, z) exp(-i (kx x + ky y))

(I didn't show the kx step very clearly)


                   ky
                   |                 G
                   |
                   |
    (Data here)    |  (Data here) 
                   |      
                   |
------------------------------------------ kx
                   |
                   |
  (Implied here)   |  (Implied here)
                   |
                   |
G*                 |




                   ky
                   |                 G
                   |
                   |
    (Data here)    |  (Data here) 
                   |      
                   |
+++++++++++++++++++O----------------------- kx
                   |
                   |
  (Implied here)   |  (Implied here)
                   |
                   |
G*                 |

kx = ky = 0 is not a mode in our simulation. 

The kx < 0 and ky = 0 modes are tricky-ish. 

Consider reality (R) for these modes: 

G(kx, ky, z) = G* (-kx, -ky, z)

If ky = 0: 

G(kx, 0, z) = G* (-kx, 0, z) <<<< enforced by "reality" function

It would be awkward to have a different number of kx entries for 
different values of ky in the array for G, so both sides of ky=0
(+/- kx) are kept, even though they are redundant. This makes accessing 
the arrays easier (they are rectangular). 

*/
Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx(pars->nx_in), // Nx is the number of real space points in the x-direction
  Ny(pars->ny_in), // Ny is the number of real space points in the y-direction
  Nz(pars->nz_in), // Nz is the number of real space points in the z-direction
  Nspecies(pars->nspec_in), // # of species (only tested for one right now)
  Nm(pars->nm_in), // number of Hermite moments
  Nl(pars->nl_in), // number of Laguerre moments (except for one choice of closure)
  // some additional derived grid sizes
  Nyc(Ny/2+1), // This is the number of unique Fourier harmonics in the y-direction
  Naky((Ny-1)/3+1), // This is the set of ky modes that are unmasked
  //Nakx(Nx - (2*Nx/3+1 - ((Nx-1)/3+1))),
  Nakx(1 + 2*((Nx-1)/3)), // This is the set of kx modes that are unmasked
  NxNyc(Nx * Nyc),
  NxNy(Nx * Ny),
  NxNycNz(Nx * Nyc * Nz),
  NxNyNz(Nx * Ny * Nz),
  NxNz(Nx * Nz),
  NycNz(Nyc * Nz),
  Nmoms(Nm*Nl),
  pars_(pars)
{
  cudaDeviceSynchronize();
  cudaMallocHost((void**) &theta0_h, sizeof(float)*Nakx);

  cudaMallocHost((void**) &kx_h, sizeof(float)*Nx); // BD why not Nakx here? (answer: linear runs?)
  cudaMallocHost((void**) &ky_h, sizeof(float)*Nyc);
  cudaMallocHost((void**) &kz_h, sizeof(float)*Nz);

  cudaMalloc((void**) &kx, sizeof(float)*Nx);
  cudaMalloc((void**) &ky, sizeof(float)*Nyc);
  cudaMalloc((void**) &kz, sizeof(float)*Nz);

  //  printf("In grids constructor. Nyc = %i \n",Nyc);
  
  // copy some parameters to device constant memory 
  cudaMemcpyToSymbol(nx,  &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny,  &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nyc, &Nyc, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz,  &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &Nspecies, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nm,  &Nm, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nl,  &Nl, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zp,  &pars_->Zp, sizeof(float),0,cudaMemcpyHostToDevice);  
  cudaMemcpyToSymbol(ikx_fixed, &pars_->ikx_fixed, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(iky_fixed, &pars_->iky_fixed, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // initialize k arrays
  int Nmax = max(max(Nx, Nyc),Nz);
  kInit<<<1, Nmax>>>(kx, ky, kz, pars_->x0, pars_->y0, pars_->Zp);

  CP_TO_CPU(kx_h, kx, sizeof(float)*Nx);
  CP_TO_CPU(ky_h, ky, sizeof(float)*Nyc);
  CP_TO_CPU(kz_h, kz, sizeof(float)*Nz);  
}

Grids::~Grids() {
  cudaFree(kx);
  cudaFree(ky);
  cudaFree(kz);
  
  cudaFreeHost(kx_h);
  cudaFreeHost(ky_h);
}


