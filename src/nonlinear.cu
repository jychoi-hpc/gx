#include "nonlinear.h"
#include "get_error.h"
#define GBK <<< dGk, dBk >>>
#define GBX <<< dGx, dBx >>>
#define GBX_single <<< dGx_single, dBx_single >>>
#define GBX_ntft <<<dGx_ntft, dBx_ntft >>>
#define GBX_single_ntft <<<dGx_single_ntft, dBx_single_ntft>>>

//===========================================
// Nonlinear_GK
// object for handling non-linear terms in GK
//===========================================
Nonlinear_GK::Nonlinear_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  red(nullptr), laguerre(nullptr), laguerre_single(nullptr)
{

  tmp_c      = nullptr;  NL_apar      = nullptr;  dG         = nullptr;  dg_dx       = nullptr;  dg_dy      = nullptr;  val1        = nullptr;
  Gy         = nullptr;  dJ0phi_dx  = nullptr;  dJ0phi_dy   = nullptr;  dJ0apar_dx = nullptr;
  dJ0apar_dy = nullptr;  dphi       = nullptr;  dchi = nullptr;  g_res       = nullptr;  
  J0phi      = nullptr;  J0apar     = nullptr;  dphi_dy     = nullptr;
  iKxG       = nullptr;  iKxG_single = nullptr;

  grad_perp_f = nullptr;
  grad_perp_G = nullptr;
  grad_perp_J0f = nullptr;
  grad_perp_G_single = nullptr;

  if (grids_ -> Nl < 2) {
    printf("\n");
    printf("Cannot do a nonlinear run with nlaguerre < 2\n");
    printf("\n");
    exit(1);
  }
  cudaStreamCreateWithFlags(&G_stream, cudaStreamDefault);
  cudaStreamCreateWithFlags(&f_stream, cudaStreamDefault);
  cudaEventCreateWithFlags(&grad_perp_f_finished, cudaEventDisableTiming);

  laguerre = new LaguerreTransform(grids_, grids_->Nm, G_stream);
  laguerre_single = new LaguerreTransform(grids_, 1);

  nBatch = grids_->Nz*grids_->Nl*grids_->Nm; 
  grad_perp_G =     new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz*grids_->Nl*grids_->Nm, G_stream); 

  nBatch = grids_->Nz*grids_->Nl; 
  grad_perp_G_single = new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz*grids_->Nl); 

  nBatch = grids_->Nz*grids_->Nj; 
  grad_perp_J0f = new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz*grids_->Nj, f_stream); 

  nBatch = grids_->Nz;
  grad_perp_f =   new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz);
  cudaDeviceSynchronize();

  checkCuda(cudaMalloc(&dG,    sizeof(float)*grids_->NxNyNz*grids_->Nl*grids_->Nm));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));

  checkCuda(cudaMalloc(&J0phi,      sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dx,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dy,  sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMemset(dG,    0., sizeof(float)*grids_->NxNyNz*grids_->Nl*grids_->Nm));
  checkCuda(cudaMemset(dg_dx, 0., sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));
  checkCuda(cudaMemset(dg_dy, 0., sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));

  checkCuda(cudaMemset(J0phi,      0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
  checkCuda(cudaMemset(dJ0phi_dx,  0., sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMemset(dJ0phi_dy,  0., sizeof(float)*grids_->NxNyNz*grids_->Nj));

  if (pars_->nonTwist) {
    checkCuda(cudaMalloc(&iKxG,     sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
    checkCuda(cudaMemset(iKxG,   0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
    if (pars_->fapar > 0.) {
      checkCuda(cudaMalloc(&iKxG_single, sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
      checkCuda(cudaMemset(iKxG_single, 0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
    }
  }

  if (pars_->fapar > 0.) {
    checkCuda(cudaMalloc(&J0apar,      sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
    checkCuda(cudaMalloc(&dJ0apar_dx,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
    checkCuda(cudaMalloc(&dJ0apar_dy,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
    checkCuda(cudaMalloc(&tmp_c,    sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));

    checkCuda(cudaMemset(J0apar,      0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
    checkCuda(cudaMemset(dJ0apar_dx,  0., sizeof(float)*grids_->NxNyNz*grids_->Nj));
    checkCuda(cudaMemset(dJ0apar_dy,  0., sizeof(float)*grids_->NxNyNz*grids_->Nj));
    checkCuda(cudaMemset(tmp_c,    0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
    NL_apar = new MomentsG(pars_, grids_);
  }

  checkCuda(cudaMalloc(&dphi,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMemset(dphi,  0., sizeof(float)*grids_->NxNyNz));
  if (pars_->fapar > 0. || pars_->fbpar > 0.) {
    checkCuda(cudaMalloc(&dchi,  sizeof(float)*grids_->NxNyNz));
    checkCuda(cudaMemset(dchi,  0., sizeof(float)*grids_->NxNyNz));
  }
  checkCuda(cudaMalloc(&g_res, sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));
  checkCuda(cudaMemset(g_res, 0., sizeof(float)*grids_->NxNyNz*grids_->Nj*grids_->Nm));

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  fXY = dphi; // this is just a pointer for use in diagnostics

  int nxyz = grids_->NxNyNz;
  int nlag = grids_->Nj;
  int nher = grids_->Nm;

  int nbx = 256;  int ngx = 1 + (grids_->NxNyNz*grids_->Nj - 1)/nbx;
  int nby = 1;  int ngy = 1;
  int nbz = 1;  int ngz = 1;
  
  // need this one to do iKx(NxNycNz) * G(NxNycNzNlNm) multiplication for NTFT
  int nbx_ntft = min(WARPSIZE, grids_->NxNycNz);  int ngx_ntft = 1 + (grids_->NxNycNz-1)/nbx_ntft;
  int nby_ntft = min(4, grids_->Nl);        int ngy_ntft = 1 + (grids_->Nl-1)/nby_ntft;

  dBx = dim3(nbx, nby, nbz);
  dGx = dim3(ngx, ngy, ngz);
  
  dBx_ntft = dim3(nbx_ntft, nby_ntft, nbz);
  dGx_ntft = dim3(ngx_ntft, ngy_ntft, ngz);

  dBx_single = dim3(nbx, nby, 1);
  dGx_single = dim3(ngx, ngy, 1);

  dBx_single_ntft= dim3(nbx_ntft, nby_ntft, 1);
  dGx_single_ntft= dim3(ngx_ntft, ngy_ntft, 1);

  int nxkyz = grids_->NxNycNz;
  
  nbx = 256;      ngx = 1 + (nxkyz-1)/nbx;
  nby = 1;       ngy = 1 + (nlag-1)/nby;

  dBk = dim3(nbx, nby, 1);
  dGk = dim3(ngx, ngy, 1);

  cfl_x_inv = (float) grids_->Nx/2 / (pars_->cfl * pars_->x0);
  cfl_y_inv = (float) grids_->Ny/2 / (pars_->cfl * pars_->y0); 
//  cfl_x_inv = (float) grids_->Nx / (pars_->cfl * 2 * M_PI * pars_->x0);
//  cfl_y_inv = (float) grids_->Ny / (pars_->cfl * 2 * M_PI * pars_->y0);
  
  dt_cfl = 0.;

  red_d_temp_storage = nullptr;
  red_temp_storage_bytes = 0;
  cub::DeviceReduce::Max(red_d_temp_storage, red_temp_storage_bytes,
                            dphi, val1, grids_->NxNyNz);
  checkCuda(cudaMalloc(&red_d_temp_storage, red_temp_storage_bytes));
  cudaDeviceSynchronize();
}

Nonlinear_GK::~Nonlinear_GK() 
{
  if ( grad_perp_G     ) delete grad_perp_G;
  if ( red             ) delete red;
  if ( laguerre        ) delete laguerre;
  if ( grad_perp_J0f ) delete grad_perp_J0f;
  if ( grad_perp_f   ) delete grad_perp_f;

  if ( tmp_c       ) cudaFree ( tmp_c       );
  if ( NL_apar       ) delete  NL_apar;
  if ( dG          ) cudaFree ( dG          );
  if ( dg_dx       ) cudaFree ( dg_dx       );
  if ( dg_dy       ) cudaFree ( dg_dy       );
  if ( val1        ) cudaFree ( val1        ); 
  if ( Gy          ) cudaFree ( Gy          );
  if ( dJ0phi_dx   ) cudaFree ( dJ0phi_dx   );
  if ( dJ0phi_dy   ) cudaFree ( dJ0phi_dy   );
  if ( dJ0apar_dx  ) cudaFree ( dJ0apar_dx  );
  if ( dJ0apar_dy  ) cudaFree ( dJ0apar_dy  );
  if ( dphi        ) cudaFree ( dphi        );
  if ( dchi        ) cudaFree ( dchi        );
  if ( g_res       ) cudaFree ( g_res       );
  if ( J0phi       ) cudaFree ( J0phi       );
  if ( J0apar      ) cudaFree ( J0apar      );
  if ( iKxG        ) cudaFree ( iKxG        );
  if ( iKxG_single ) cudaFree ( iKxG_single );
  if ( red_d_temp_storage ) cudaFree (red_d_temp_storage);
  cudaStreamDestroy(f_stream);
  cudaStreamDestroy(G_stream);
}

void Nonlinear_GK::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void Nonlinear_GK::qvar (float* G, int N)
{
  float* G_h;
  int N_ = grids_->Ny*grids_->Nx*grids_->Nz;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) {G_h[i] = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = %e \n", i%N_, i/N_, G_h[i]);
  printf("\n");

  free (G_h);
}

void Nonlinear_GK::nlps(MomentsG* G, Fields* f, MomentsG* G_res)
{
  // BD  J0fToGrid does not use a Laguerre transform. Implications?
  // BD  If we use alternate forms for <J0> then that would need to be reflected here

  float rho2s = G->species->rho2;
  float vts   = G->species->vt;
  float tz    = G->species->tz;

  // compute dG_m/dx(x,y,z,mu)
  if (pars_->nonTwist) {
    iKxgtoGrid GBX_ntft (iKxG, G->G(), grids_->iKx, false);
    grad_perp_G -> C2R(iKxG, dG);
  } else {
    grad_perp_G -> dxC2R(G->G(), dG);
  }
  if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_G->phase_mult(dG, pars_->nonTwist, pars_->ExBshear_phase);
  laguerre    -> transformToGrid(dG, dg_dx);
     
  // compute dG_m/dy(x,y,z,mu)
  grad_perp_G -> dyC2R(G->G(), dG);      
  if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_G->phase_mult(dG, pars_->nonTwist, pars_->ExBshear_phase);
  laguerre    -> transformToGrid(dG, dg_dy);

  if (pars_->fapar > 0.) {
    // compute J0 Apar on mu grid
    J0fToGrid <<< dGk, dBk, 0, f_stream >>>  (J0apar, f->apar, geo_->kperp2, laguerre->get_roots(), rho2s, pars_->fapar);
   
    // compute d(J0 Apar)/dx
    if (pars_->nonTwist) {
      iKxJ0ftoGrid GBK (J0apar, J0apar, grids_->iKx, false);
      grad_perp_J0f -> C2R(J0apar, dJ0apar_dx);
    } else {
      grad_perp_J0f -> dxC2R(J0apar, dJ0apar_dx);
    }
    if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_J0f -> phase_mult(dJ0apar_dx, pars_->nonTwist, pars_->ExBshear_phase);

    // compute d(J0 Apar)/dy
    grad_perp_J0f -> dyC2R(J0apar, dJ0apar_dy);
    if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_J0f -> phase_mult(dJ0apar_dy, pars_->nonTwist, pars_->ExBshear_phase);

    // compute {G_m, J0 Apar}
    checkCuda(cudaEventRecord(grad_perp_f_finished, f_stream));
    cudaStreamWaitEvent(G_stream, grad_perp_f_finished);
    bracket <<< dGx, dBx, 0, G_stream >>> (g_res, dg_dx, dJ0apar_dy, dg_dy, dJ0apar_dx, pars_->kxfac);

    laguerre->transformToSpectral(g_res, dG);
    if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_G -> phase_mult(dG, pars_->nonTwist, pars_->ExBshear_phase, false);
    grad_perp_G->R2C(dG, NL_apar->G(), false); // this R2C has accumulate=false
    NL_apar->sync(false, 1); // non-blocking sync, only m+/-1
  }
  
  // Note that when parallelizing over m, Nm = Nm/nprocs_m < total Nm
  // so that if the input Nm = 16 but each node has four Hermites (for example),
  // then here Nm = 4.
  // On the other hand, there are ghost cells required when fapar = 1. (finite A_parallel) 
  //
  // loop over m to save memory. also makes it easier to parallelize.
  // no extra computation: just no batching in m in FFTs and in the matrix multiplies
  
  if(pars_->fbpar > 0.0) {
    J0phiAndBparToGrid <<< dGk, dBk, 0, f_stream >>> (J0phi, f->phi, f->bpar, geo_->kperp2, laguerre->get_roots(), rho2s, tz, pars_->fphi, pars_->fbpar);
  } else {
    J0fToGrid <<< dGk, dBk, 0, f_stream >>> (J0phi, f->phi, geo_->kperp2, laguerre->get_roots(), rho2s, pars_->fphi);
  }

  // JMH d/dx->ikx operator in iKxtoGrid functions for NTFT (iKx) or NTFT + ExB (iKxstar = iKx - ky*g_exb*dt), within dxC2R callback if ExB only (ikxstar)  
  // phasefactor in 1D fft within phase_mult, called once per d/dx or d/dy parameter if NTFT, NTFT+ExB, or ExB
  // iKx multiplication is in place except for G/Gsingle so we don't overwrite things used elsewhere, J0phi/apar are used in this function only
  if (pars_->nonTwist) { // d/dx and positive exponential phase factor calulation
    iKxJ0ftoGrid GBK (J0phi, J0phi, grids_->iKx, false);
    grad_perp_J0f -> C2R(J0phi, dJ0phi_dx);
  } else { //perform d/dx as normal for conventional flux tube 
    grad_perp_J0f -> dxC2R(J0phi, dJ0phi_dx);
  }
  if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_J0f -> phase_mult(dJ0phi_dx, pars_->nonTwist, pars_->ExBshear_phase);
  
  grad_perp_J0f -> dyC2R(J0phi, dJ0phi_dy);
  if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_J0f -> phase_mult(dJ0phi_dy, pars_->nonTwist, pars_->ExBshear_phase);
  
  // compute {G_m, J0 chi}
  checkCuda(cudaEventRecord(grad_perp_f_finished, f_stream));
  cudaStreamWaitEvent(G_stream, grad_perp_f_finished);
  bracket <<< dGx, dBx, 0, G_stream >>> (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);
  laguerre->transformToSpectral(g_res, dG);
  if (pars_->nonTwist || pars_->ExBshear_phase) grad_perp_G -> phase_mult(dG, pars_->nonTwist, pars_->ExBshear_phase, false);
  // NL_m += {G_m, J0 chi}
  grad_perp_G->R2C(dG, G_res->G(), true); // this R2C has accumulate=true

  // finish Apar NL term after ghost sync completes
  if (pars_->fapar > 0.) {
    if(grids_->nprocs_m > 1) cudaStreamWaitEvent(G_stream, NL_apar->finished_sync);

    // NL_m += -vt*sqrt(m)*{G_{m-1}, J0 Apar} - vt*sqrt(m+1)*{G_{m+1}, J0 Apar} 
    // NL_apar(m) == {G_m, J0 Apar}, so
    // NL_m += -vt*sqrt(m)*NL_apar(m-1) - vt*sqrt(m+1)*NL_apar(m+1)
    nl_flutter <<< dGk, dBk, 0, G_stream >>> (G_res->G(), NL_apar->G(), vts);
  }
}

void Nonlinear_GK::get_max_frequency(Fields *f, double *omega_max)
{
  float vpar_max = grids_->vpar_max*pars_->vtmax; // estimate of max vpar on grid
  float muB_max = grids_->muB_max*pars_->tzmax; 
  grad_perp_f -> dxC2R(f->phi, dphi); 
  abs <<<dGx.x,dBx.x>>> (dphi, grids_->NxNyNz);

  if(pars_->fapar > 0.0) {
    grad_perp_f -> dxC2R(f->apar, dchi); 
    abs <<<dGx.x,dBx.x>>> (dchi, grids_->NxNyNz);
    add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi, 1., dphi, vpar_max, dchi);
  }
  if(pars_->fbpar > 0.0) {
    grad_perp_f -> dxC2R(f->bpar, dchi); 
    abs <<<dGx.x,dBx.x>>> (dchi, grids_->NxNyNz);
    add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi, 1., dphi, muB_max, dchi);
  }
  cub::DeviceReduce::Max(red_d_temp_storage, red_temp_storage_bytes,
                            dphi, val1, grids_->NxNyNz);
  CP_TO_CPU(vmax_y, val1, sizeof(float));

  grad_perp_f -> dyC2R(f->phi, dphi);  
  abs <<<dGx.x,dBx.x>>> (dphi, grids_->NxNyNz);
  if(pars_->fapar > 0.0) {
    grad_perp_f -> dyC2R(f->apar, dchi); 
    abs <<<dGx.x,dBx.x>>> (dchi, grids_->NxNyNz);
    add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi, 1., dphi, vpar_max, dchi);
  }
  if(pars_->fbpar > 0.0) {
    grad_perp_f -> dyC2R(f->bpar, dchi); 
    abs <<<dGx.x,dBx.x>>> (dchi, grids_->NxNyNz);
    add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi, 1., dphi, muB_max, dchi);
  }
  cub::DeviceReduce::Max(red_d_temp_storage, red_temp_storage_bytes,
                            dphi, val1, grids_->NxNyNz);
  CP_TO_CPU(vmax_x, val1, sizeof(float));
  double scale = 0.5;  // normalization scaling factor for C2R FFT
  omega_max[0] = fmax(omega_max[0], fabs(pars_->kxfac)*(grids_->kx_max*vmax_x[0])*scale);
  omega_max[1] = fmax(omega_max[1], fabs(pars_->kxfac)*(grids_->ky_max*vmax_y[0])*scale);
}

//==============================================
// Nonlinear_KREHM
// object for handling non-linear terms in KREHM
//==============================================
Nonlinear_KREHM::Nonlinear_KREHM(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), red(nullptr)
{
  tmp_c = nullptr;
  dG = nullptr;
  dg_dx = nullptr;
  dg_dy = nullptr;
  dphi_dx = nullptr;
  dphi_dy = nullptr;
  dchi_dx = nullptr;
  dchi_dy = nullptr;
  G_tmp = nullptr;

  grad_perp_f = grad_perp_G = nullptr;

  nBatch = grids_->Nz*grids_->Nm; 
  grad_perp_G =     new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz*grids_->Nm); 

  nBatch = grids_->Nz; 
  grad_perp_f = new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz);

  red = new Reduction<float>(grids_, {'r', 'x', 'z'}, {}); 
  cudaDeviceSynchronize();

  G_tmp = new MomentsG(pars_, grids_);

  checkCuda(cudaMalloc(&tmp_c, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc(&dG,    sizeof(float)*grids_->NxNyNz*grids_->Nm));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*grids_->NxNyNz*grids_->Nm));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->NxNyNz*grids_->Nm));

  checkCuda(cudaMalloc(&dphi_dx,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dphi_dy,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dchi_dx,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dchi_dy,  sizeof(float)*grids_->NxNyNz));

  checkCuda(cudaMemset(tmp_c, 0., sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMemset(dG,    0., sizeof(float)*grids_->NxNyNz*grids_->Nm));
  checkCuda(cudaMemset(dg_dx, 0., sizeof(float)*grids_->NxNyNz*grids_->Nm));
  checkCuda(cudaMemset(dg_dy, 0., sizeof(float)*grids_->NxNyNz*grids_->Nm));

  checkCuda(cudaMemset(dphi_dx,  0., sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMemset(dphi_dy,  0., sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMemset(dchi_dx,  0., sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMemset(dchi_dy,  0., sizeof(float)*grids_->NxNyNz));

  fXY = dphi_dx; // this is just a pointer for use in diagnostics

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  int nxyz = grids_->NxNyNz;
  int nher = grids_->Nm;

  int nbx = min(WARPSIZE, nxyz);  int ngx = 1 + (nxyz-1)/nbx; 
  int nby = 1;  int ngy = 1;  // no laguerre in KREHM
  int nbz = min(4, nher);  int ngz = 1 + (nher-1)/nbz;

  dBx = dim3(nbx, nby, nbz);
  dGx = dim3(ngx, ngy, ngz);

  dBx_single = dim3(nbx, 1, 1);
  dGx_single = dim3(ngx, 1, 1);

  int nxkyz = grids_->NxNycNz;
  
  nbx = min(WARPSIZE, nxkyz);      ngx = 1 + (nxkyz-1)/nbx;

  dBk = dim3(nbx, 1, 1);
  dGk = dim3(ngx, 1, 1);

  cfl_x_inv = (float) grids_->Nx / (pars_->cfl * 2 * M_PI * pars_->x0);
  cfl_y_inv = (float) grids_->Ny / (pars_->cfl * 2 * M_PI * pars_->y0); 
  
  dt_cfl = 0.;

  rho_s = pars->rho_s;
  d_e = pars->d_e;
}

Nonlinear_KREHM::~Nonlinear_KREHM() 
{
  if ( grad_perp_f ) delete grad_perp_f;
  if ( grad_perp_G ) delete grad_perp_G;
  if ( dg_dx ) cudaFree ( dg_dx );
  if ( dg_dy ) cudaFree ( dg_dy );
  if ( dG ) cudaFree ( dG );
  if ( tmp_c ) cudaFree ( tmp_c );
  if ( dphi_dx ) cudaFree ( dphi_dx );
  if ( dphi_dy ) cudaFree ( dphi_dy );
  if ( dchi_dx ) cudaFree ( dchi_dx );
  if ( dchi_dy ) cudaFree ( dchi_dy );
  if ( val1 ) cudaFree ( val1 ); 
  if ( red ) delete red;
}

void Nonlinear_KREHM::nlps(MomentsG* G, Fields* f, MomentsG* G_nl)
{
  grad_perp_f->dxC2R(f->phi, dphi_dx);
  grad_perp_f->dyC2R(f->phi, dphi_dy);
  grad_perp_f->dxC2R(f->apar, dchi_dx);
  grad_perp_f->dyC2R(f->apar, dchi_dy);

  grad_perp_G->dxC2R(G->G(), dg_dx);
  grad_perp_G->dyC2R(G->G(), dg_dy);      

  // compute {g_m, phi}
  bracket GBX (dG, dg_dx, dphi_dy, dg_dy, dphi_dx, 1.);
  // NL_m += {g_m, phi}
  grad_perp_G->R2C(dG, G_nl->G(), true); // this R2C has accumulate=true

  // compute {g_m, Apar}
  bracket GBX (dG, dg_dx, dchi_dy, dg_dy, dchi_dx, 1.);
  grad_perp_G->R2C(dG, G_tmp->G(), false); // this R2C has accumulate=false

  for(int m=grids_->m_lo; m<grids_->m_up; m++) {
    int m_local = m - grids_->m_lo;

    // NL_{m+1} += -rho_s/d_e*sqrt(m+1)*{g_m, Apar}
    if(m+1 < pars_->nm_in) add_scaled_singlemom_kernel <<<dGk.x,dBk.x>>> (G_nl->Gm(m_local+1), 1., G_nl->Gm(m_local+1), -rho_s/d_e*sqrtf(m+1), G_tmp->Gm(m_local));
    // NL_{m-1} += -rho_s/d_e*sqrt(m)*{G_m, Apar}
    if(m>0) add_scaled_singlemom_kernel <<<dGk.x,dBk.x>>> (G_nl->Gm(m_local-1), 1., G_nl->Gm(m_local-1), -rho_s/d_e*sqrtf(m), G_tmp->Gm(m_local));
  }

  // contributions from ghost cells (EM only)
  if(grids_->nprocs_m>1) {
    cudaStreamSynchronize(G->syncStream);

    // lower ghost
    int m = grids_->m_lo;
    int m_local = m - grids_->m_lo;
    if(m>0) {
      grad_perp_f -> dxC2R(G->Gm(m_local-1), dg_dx);
      grad_perp_f -> dyC2R(G->Gm(m_local-1), dg_dy);      
      bracket GBX_single (dG, dg_dx, dchi_dy, dg_dy, dchi_dx, 1.0);
      grad_perp_f->R2C(dG, tmp_c, false); // this R2C has accumulate=false
      // NL_{m} += -rho_s/d_e*sqrt(m)*{G_{m-1}, Apar}
      add_scaled_singlemom_kernel <<<dGk.x,dBk.x>>> (G_nl->Gm(m_local), 1., G_nl->Gm(m_local), -rho_s/d_e*sqrtf(m), tmp_c);
    }

    // upper ghost
    m = grids_->m_up-1;
    m_local = m - grids_->m_lo;
    if(m<pars_->nm_in-1) {
      grad_perp_f -> dxC2R(G->Gm(m_local+1), dg_dx);
      grad_perp_f -> dyC2R(G->Gm(m_local+1), dg_dy);
      bracket GBX_single (dG, dg_dx, dchi_dy, dg_dy, dchi_dx, 1.0);
      grad_perp_f->R2C(dG, tmp_c, false); // this R2C has accumulate=false
      // NL_{m} += -rho_s/d_e*sqrt(m+1)*{G_{m+1}, Apar}
      add_scaled_singlemom_kernel <<<dGk.x,dBk.x>>> (G_nl->Gm(m_local), 1., G_nl->Gm(m_local), -rho_s/d_e*sqrtf(m+1), tmp_c);
    }
  }
}

void Nonlinear_KREHM::get_max_frequency(Fields *f, double *omega_max)
{
  float vpar_max = grids_->vpar_max*rho_s/d_e; // estimate of max vpar on grid

  grad_perp_f -> dxC2R(f->phi, dphi_dx); 
  abs <<<dGx.x,dBx.x>>> (dphi_dx, grids_->NxNyNz);
  grad_perp_f -> dxC2R(f->apar, dchi_dx); 
  abs <<<dGx.x,dBx.x>>> (dchi_dx, grids_->NxNyNz);
  add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi_dx, 1., dphi_dx, vpar_max, dchi_dx);
  //printf("val1 = %lf, dphi_dx = %lf\n", val1, dphi_dx);
  red->Max(dphi_dx, val1); 
  CP_TO_CPU(vmax_y, val1, sizeof(float));

  grad_perp_f -> dyC2R(f->phi, dphi_dy);  
  abs <<<dGx.x,dBx.x>>> (dphi_dy, grids_->NxNyNz);
  grad_perp_f -> dyC2R(f->apar, dchi_dy); 
  abs <<<dGx.x,dBx.x>>> (dchi_dy, grids_->NxNyNz);
  add_scaled_singlemom_kernel <<<dGx.x,dBx.x>>> (dphi_dy, 1., dphi_dy, vpar_max, dchi_dy);
  //printf("val1 = %lf, dphi_dy = %lf\n", val1, dphi_dy);
  red->Max(dphi_dy, val1); 
  CP_TO_CPU(vmax_x, val1, sizeof(float));
  //printf("vpar_max = %lf\n\n", vpar_max);
  double scale = 1.0;  // normalization scaling factor for C2R FFT
  omega_max[0] = fmax(omega_max[0], (grids_->kx_max*vmax_x[0])*scale);
  omega_max[1] = fmax(omega_max[1], (grids_->ky_max*vmax_y[0])*scale);
  //printf("omega_max[0]= %lf \n omega_max[1]= %lf\n", omega_max[0], omega_max[1]);
  //printf("kx_max*vmax_x[0] = %lf \n  grids_->ky_max*vmax_x[0] = %lf \n\n", (grids_->kx_max*vmax_x[0])*scale, (grids_->ky_max*vmax_y[0])*scale);
}

//==============================================
// Nonlinear_cetg
// object for handling non-linear terms in cetg
//==============================================
Nonlinear_cetg::Nonlinear_cetg(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), red(nullptr)
{
  dG = nullptr;
  dg_dx = nullptr;
  dg_dy = nullptr;
  dphi_dx = nullptr;
  dphi_dy = nullptr;

  nBatch = 2*grids_->Nz; 
  grad_perp_G =     new GradPerp(pars_, grids_, nBatch, 2 * grids_->NxNycNz); 

  nBatch = grids_->Nz; 
  grad_perp_f = new GradPerp(pars_, grids_, nBatch, grids_->NxNycNz);

  std::vector<int32_t> modes{'r', 'x', 'z'};
  std::vector<int32_t> modesRed{};
  red = new Reduction<float>(grids_, modes, modesRed); 

  checkCuda(cudaMalloc(&dG,    sizeof(float)*2*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*2*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*2*grids_->NxNyNz));

  checkCuda(cudaMalloc(&dphi_dx,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dphi_dy,  sizeof(float)*grids_->NxNyNz));

  checkCuda(cudaMemset(dG,    0., sizeof(float)*2*grids_->NxNyNz));
  checkCuda(cudaMemset(dg_dx, 0., sizeof(float)*2*grids_->NxNyNz));
  checkCuda(cudaMemset(dg_dy, 0., sizeof(float)*2*grids_->NxNyNz));

  checkCuda(cudaMemset(dphi_dx,  0., sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMemset(dphi_dy,  0., sizeof(float)*grids_->NxNyNz));

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  int nxyz = grids_->NxNyNz;

  int nbx = min(256, nxyz);  int ngx = 1 + (nxyz-1)/nbx; 
  int nby = 1;               int ngy = 1; 
  int nbz = 1;               int ngz = 1;  

  dBx = dim3(nbx, nby, nbz);
  dGx = dim3(ngx, ngy, ngz);

  dBx_single = dim3(nbx, 1, 1);
  dGx_single = dim3(ngx, 1, 1);

  int nxkyz = grids_->NxNycNz;
  
  nbx = min(128, nxkyz);      ngx = 1 + (nxkyz-1)/nbx;

  dBk = dim3(nbx, 1, 1);
  dGk = dim3(ngx, 1, 1);

  cfl_x_inv = (float) grids_->Nx / (pars_->cfl * 2 * M_PI * pars_->x0);
  cfl_y_inv = (float) grids_->Ny / (pars_->cfl * 2 * M_PI * pars_->y0); 
  
  dt_cfl = 0.;

}

Nonlinear_cetg::~Nonlinear_cetg() 
{
  if ( grad_perp_f ) delete grad_perp_f;
  if ( grad_perp_G ) delete grad_perp_G;
  if ( dg_dx )   cudaFree ( dg_dx );
  if ( dg_dy )   cudaFree ( dg_dy );
  if ( dG )      cudaFree ( dG );
  if ( dphi_dx ) cudaFree ( dphi_dx );
  if ( dphi_dy ) cudaFree ( dphi_dy );
  if ( val1 )    cudaFree ( val1 ); 
  if ( red ) delete red;
}

void Nonlinear_cetg::nlps(MomentsG* G, Fields* f, MomentsG* G_nl)
{
  grad_perp_f->dxC2R(f->phi, dphi_dx);
  grad_perp_f->dyC2R(f->phi, dphi_dy);

  grad_perp_G->dxC2R(G->G(), dg_dx);
  grad_perp_G->dyC2R(G->G(), dg_dy);      

  // compute {g, phi} / 2.
  bracket_cetg GBX (dG, dg_dx, dphi_dy, dg_dy, dphi_dx, 0.5);
  // NL = {g, phi} / 2. 
  grad_perp_G->R2C(dG, G_nl->G(), false); // this R2C has accumulate=false

}

void Nonlinear_cetg::get_max_frequency(Fields *f, double *omega_max)
{

  grad_perp_f -> dxC2R(f->phi, dphi_dx); 
  abs <<<dGx.x,dBx.x>>> (dphi_dx, grids_->NxNyNz);
  red->Max(dphi_dx, val1); 
  CP_TO_CPU(vmax_y, val1, sizeof(float));

  grad_perp_f -> dyC2R(f->phi, dphi_dy);  
  abs <<<dGx.x,dBx.x>>> (dphi_dy, grids_->NxNyNz);
  red->Max(dphi_dy, val1); 
  CP_TO_CPU(vmax_x, val1, sizeof(float));

  double scale = 1.0;  // normalization scaling factor for C2R FFT
  omega_max[0] = fmax(omega_max[0], (grids_->kx_max*vmax_x[0])*scale);
  omega_max[1] = fmax(omega_max[1], (grids_->ky_max*vmax_y[0])*scale);
}


