#include "grids.h"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx       ( pars->nx_in       ),
  Ny       ( pars->ny_in       ),
  Nz       ( pars->nz_in       ),
  Nspecies ( pars->nspec_in    ),
  Nm       ( pars->nm_in       ),
  Nl       ( pars->nl_in       ),
  Nj       ( 3*pars->nl_in/2-1 ),

  Nyc      ( 1 + Ny/2          ),
  Naky     ( 1 + (Ny-1)/3      ),
  Nakx     ( 1 + 2*((Nx-1)/3)  ), 
  NxNyc    ( Nx * Nyc          ),
  NxNy     ( Nx * Ny           ),
  NxNycNz  ( Nx * Nyc * Nz     ),
  NxNyNz   ( Nx * Ny * Nz      ),
  NxNz     ( Nx * Nz           ),
  NycNz    ( Nyc * Nz          ),
  Nmoms    ( Nm * Nl           ),
  size_G( sizeof(cuComplex) * NxNycNz * Nmoms), 
  pars_(pars)
{
  ky              = nullptr;  kx              = nullptr;  kz              = nullptr;
  ky_h            = nullptr;  kx_h            = nullptr;  kz_h            = nullptr;
  kx_outh         = nullptr;
  kz_outh         = nullptr;  kpar_outh       = nullptr;  kzp             = nullptr;
  y_h             = nullptr;  kxs             = nullptr;  x_h             = nullptr;
  theta0_h        = nullptr;  th0             = nullptr;  z_h             = nullptr;
  
  // kz is defined without the factor of gradpar
  
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  kx_outh   = (float*) malloc(sizeof(float) * Nakx       );
  cudaMalloc     ( (void**) &kzm,       sizeof(int)   * Nz       );
  cudaMalloc     ( (void**) &kzp,       sizeof(float) * Nz       );
  kz_outh   = (float*) malloc(sizeof(float) * Nz       );
  kpar_outh = (float*) malloc(sizeof(float) * Nz       );
  theta0_h  = (float*) malloc(sizeof(float) * Nx       ); 
  kx_h      = (float*) malloc(sizeof(float) * Nx       ); 
  ky_h      = (float*) malloc(sizeof(float) * Nyc      );
  kz_h      = (float*) malloc(sizeof(float) * Nz       );
  cudaMalloc     ( (void**) &kx,        sizeof(float) * Nx       );
  cudaMalloc     ( (void**) &th0,       sizeof(float) * Nx       );
  cudaMalloc     ( (void**) &ky,        sizeof(float) * Nyc      );
  cudaMalloc     ( (void**) &kz,        sizeof(float) * Nz       );
  x_h      = (float*) malloc(sizeof(float) * Nx       ); 
  y_h      = (float*) malloc(sizeof(float) * Ny       );
  z_h      = (float*) malloc(sizeof(float) * Nz       );
  cudaMalloc     ( (void**) &kxs,       sizeof(float) * Nx * Nyc );
  checkCuda(cudaGetLastError());

  //  printf("In grids constructor. Nyc = %i \n",Nyc);
  
  setdev_constants(Nx, Ny, Nyc, Nz, Nspecies, Nm, Nl, Nj, pars_->Zp, pars_->ikx_fixed, pars_->iky_fixed);

  checkCuda(cudaDeviceSynchronize());

  // initialize k arrays
  int Nmax = max(max(Nx, Nyc), Nz);
  int nt = min(32, Nmax);
  int nb = 1 + (Nmax-1)/nt;

  kInit <<<nb, nt>>> (kx, ky, kz, kzm, kzp, pars_->x0, pars_->y0, pars_->Zp, pars_->dealias_kz);

  CP_TO_CPU (kx_h, kx, sizeof(float)*Nx);
  CP_TO_CPU (ky_h, ky, sizeof(float)*Nyc);
  CP_TO_CPU (kz_h, kz, sizeof(float)*Nz);

  // If this is a restarted run, should get kxs from the restart file
  // otherwise:
  if (pars_->ExBshear) {
    int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;
    nn1 = Nyc;       nt1 = min(32, nn1);     nb1 = 1 + (nn1-1)/nt1;
    nn2 = Nx;        nt2 = min(32, nn2);     nb2 = 1 + (nn2-1)/nt2;
    nn3 = 1;         nt3 = 1;                nb3 = 1;
    dim3 dB = (nt1, nt2, nt3);
    dim3 dG = (nb1, nb2, nb3);
    init_kxs <<< dG, dB >>> (kxs, kx, th0);
    CP_TO_CPU (theta0_h, th0, sizeof(float)*Nx);    
  }
  
  if (Nx<4) {
    //    printf("Nx, Nakx = %d, %d \n",Nx, Nakx);
    //    printf("kx_h = %f \n",kx_h[0]);
    for (int i=0; i<Nx; i++) kx_outh[i] = kx_h[i];
  } else {    
    kx_outh[Nakx/2] = 0.;
    for (int i=1; i<Nakx/2+1; i++) {
      kx_outh[Nakx/2 + i] = kx_h[i];
      kx_outh[Nakx/2 - i] = kx_h[Nx-i];
    }
    /*
    for (int i=0; i<Nx; i++) {
      printf("kx_h[%d] = %f \t",i,kx_h[i]);
    }
    printf("\n");
    for (int i = 0; i < Nakx; i++) {
      printf("kx_outh[%d] = %f \t",i, kx_outh[i]);
    }
    */
  }

  if (Nz>1) {
    for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ (i + Nz/2 + 1) % Nz ];
  } else {
    for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ i ];
  }
  
  // define the y coordinate
  y_h[0] = 0.;
  for (int i = 1; i < Ny ; i++) y_h[i] = y_h[i-1] + (float) 2*M_PI*(pars_->y0)/Ny;

  // Could define a variable that keeps track of y(t) when there is ExB shear but it can be derived from what is
  // already written
  
  // define the x coordinate
  x_h[0] = 0.;
  for (int i = 1; i < Nx ; i++) x_h[i] = x_h[i-1] + (float) 2*M_PI*(pars_->x0)/Nx;

  // define the z coordinate
  for(int k=0; k<Nz; k++) {
    z_h[k] = 2.*M_PI *pars->Zp *(k-Nz/2)/Nz;
  }
}

Grids::~Grids() {
  if (kxs)             cudaFree(kxs);
  if (kx)              cudaFree(kx);
  if (ky)              cudaFree(ky);
  if (kz)              cudaFree(kz);
  if (kzm)             cudaFree(kzm);
  if (kzp)             cudaFree(kzp);
  if (th0)             cudaFree(th0);
  
  if (kpar_outh)       free(kpar_outh);
  if (kz_outh)         free(kz_outh);
  if (kx_outh)         free(kx_outh);
  if (kx_h)            free(kx_h);
  if (ky_h)            free(ky_h);
  if (kz_h)            free(kz_h);
  if (x_h)             free(x_h);
  if (y_h)             free(y_h);
  if (z_h)             free(z_h);
  if (theta0_h)        free(theta0_h); 
 
}


