#include "grids.h"
#include "hermite_transform.h"
#include "laguerre_transform.h"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx       ( pars->nx_in       ),
  Ny       ( pars->ny_in       ),
  Nz       ( pars->nz_in       ),
  Nl       ( pars->nl_in       ),
  Nj       ( max(1, 3*pars->nl_in/2-1) ),

  Nyc      ( 1 + Ny/2          ),
  Naky     ( 1 + (Ny-1)/3      ),
  Nakx     ( 1 + 2*((Nx-1)/3)  ), 
  NxNyc    ( Nx * Nyc          ),
  NxNy     ( Nx * Ny           ),
  NxNycNz  ( Nx * Nyc * Nz     ),
  NxNyNz   ( Nx * Ny * Nz      ),
  NxNz     ( Nx * Nz           ),
  NycNz    ( Nyc * Nz          ),
  Zp(pars->Zp),
  iproc(pars->iproc),
  nprocs(pars->nprocs),
  pars_(pars)
{
  ky              = nullptr;  kx              = nullptr;  kz              = nullptr;
  ky_h            = nullptr;  kx_h            = nullptr;  kz_h            = nullptr;
  kx_outh         = nullptr;
  kz_outh         = nullptr;  kpar_outh       = nullptr;  kzp             = nullptr;
  y_h             = nullptr;  kxs             = nullptr;  x_h             = nullptr;
  theta0_h        = nullptr;  th0             = nullptr;  z_h             = nullptr;

  Nspecies = pars->nspec_in;
  Nspecies_glob = Nspecies;
  Nm = pars->nm_in;
  Nm_glob = pars->nm_in;
  is_lo = 0;
  is_up = Nspecies;
  m_lo = 0;
  m_up = Nm;
  m_ghost = 0;
  nprocs_s = 1;
  nprocs_m = 1;
  iproc_m = 0;
  iproc_s = 0;

  // compute parallel decomposition
  if(nprocs>1) {
    // prioritize species decomp
    if(nprocs<=Nspecies) {
      assert((Nspecies%nprocs == 0) && "nprocs <= nspecies, so nspecies must be an integer multiple of nprocs\n");
      // this is now the local Nspecies on this proc
      Nspecies = Nspecies/nprocs;
      nprocs_s = nprocs;
      nprocs_m = 1;
      iproc_s = iproc;
      iproc_m = 0;
      is_lo = iproc*Nspecies;
      is_up = (iproc+1)*Nspecies;

      m_lo = 0;
      m_up = Nm;

      //printf("GPU %d: is_lo = %d, is_up = %d, m_lo = %d, m_up = %d\n", iproc, is_lo, is_up, m_lo, m_up);
    } else { // decomp in species and hermite
      assert((nprocs%Nspecies == 0) && "nprocs > nspecies, so nprocs must be an integer multiple of nspecies\n");
      nprocs_s = Nspecies;
      nprocs_m = nprocs/Nspecies;
      iproc_s = iproc/nprocs_m;
      iproc_m = iproc%nprocs_m;

      // this is now the local Nspecies on this proc
      Nspecies = 1;
      is_lo = iproc_s*Nspecies;
      is_up = (iproc_s+1)*Nspecies; // is_up is never used

      assert((Nm%nprocs_m == 0) && "Nm must be an integer multiple of nprocs_m=nprocs/nspecies\n");
      // this is now the local Nm on this proc
      Nm = Nm/nprocs_m;

      m_lo = (iproc_m    )*Nm;
      m_up = (iproc_m + 1)*Nm;

      // add ghosts in m
      if(pars->slab && Nm>1) {
        m_ghost = 1;
      } else {
        m_ghost = 2;
      }
    }
  }

  //
  // When solving Toby's collisional slab ETG model, use nhermite = 1, nlaguerre = 2
  // The zeroth moment will be (m=0, l=0) == Phi
  // The first moment will be (m=0, l=1) == delta T
  // These values are automatically set in parameters when this equation set is selected.

  // Should add an assert statement here? Something like "if we are solving cetg, assert Nm == 1)" and so forth
  
  Nmoms = Nm * Nl;
  size_G = sizeof(cuComplex) * NxNycNz * (Nm + 2*m_ghost) * Nl; // this includes ghosts on either end of m grid
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
  
  setdev_constants(Nx, Ny, Nyc, Nz, Nspecies, Nm, Nl, Nj, pars_->Zp, pars_->ikx_fixed, pars_->iky_fixed, is_lo, is_up, m_lo, m_up, m_ghost, pars_->nm_in);

  checkCuda(cudaDeviceSynchronize());

  DEBUGPRINT("Initializing NCCL comms...\n");
  //cudaStreamCreate(&ncclStream);
  if(iproc == 0) ncclGetUniqueId(&ncclId);
  if(nprocs > 1) {
    MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  // set up some additional ncclIds
  if(iproc == 0) ncclGetUniqueId(&ncclId_m0);
  if(nprocs > 1) {
    MPI_Bcast((void *)&ncclId_m0, sizeof(ncclId_m0), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  ncclId_s.resize(nprocs_s);
  for(int i=0; i<nprocs_s; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_s[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_s[i], sizeof(ncclId_s[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }
  ncclId_m.resize(nprocs_m);
  for(int i=0; i<nprocs_m; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_m[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_m[i], sizeof(ncclId_m[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }

  DEBUGPRINT("Got NCCL IDs\n");

  checkCuda(ncclCommInitRank(&ncclComm, nprocs, ncclId, iproc));
  // set up NCCL communicator that is per-species
  checkCuda(ncclCommInitRank(&ncclComm_s, nprocs_m, ncclId_s[iproc_s], iproc_m));
  // set up NCCL communicator that is per-m block
  checkCuda(ncclCommInitRank(&ncclComm_m, nprocs_s, ncclId_m[iproc_m], iproc_s));
  // set up NCCL communicator that involves only GPUs containing m=0, i.e. grids_->proc(0, iproc_s)
  if(iproc_m == 0) {
    if(nprocs_m > 1)
      checkCuda(ncclCommInitRank(&ncclComm_m0, nprocs_s, ncclId_m0, iproc_s));
    else
      ncclComm_m0 = ncclComm;
  }
  DEBUGPRINT("Finished initializaing NCCL comms.\n");
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
 
  ncclCommDestroy(ncclComm);
  ncclCommDestroy(ncclComm_s);
  ncclCommDestroy(ncclComm_m);
  if(nprocs_m > 1 && iproc_m == 0) ncclCommDestroy(ncclComm_m0);
}

void Grids::init_ks_and_coords()
{
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
    z_h[k] = 2.*M_PI *pars_->Zp *(k-Nz/2)/Nz;
  }

  HermiteTransform * hermite = new HermiteTransform(this);
  LaguerreTransform * laguerre = new LaguerreTransform(this, 1);
  vpar_max = hermite->get_vmax();
  muB_max = laguerre->get_vmax();
  kx_max = kx_h[(Nx-1)/3];
  ky_max = ky_h[(Ny-1)/3];
  kz_max = kz_h[Nz/2];
  kperp_min = min(kx_h[1], ky_h[1]);
  delete hermite;
  delete laguerre;
}
