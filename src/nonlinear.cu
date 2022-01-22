#include "nonlinear.h"
#include "get_error.h"
#define GBK <<< dGk, dBk >>>
#define GBX <<< dGx, dBx >>>

//===========================================
// Nonlinear_GK
// object for handling non-linear terms in GK
//===========================================
Nonlinear_GK::Nonlinear_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  red(nullptr), laguerre(nullptr), grad_perp_G(nullptr), grad_perp_J0phi(nullptr), grad_perp_phi(nullptr)
{

  dG          = nullptr;  dg_dx       = nullptr;  dg_dy       = nullptr;  val1        = nullptr;
  Gy          = nullptr;  dJ0phi_dx   = nullptr;  dJ0phi_dy   = nullptr;  dJ0_Apar_dx = nullptr;
  dJ0_Apar_dy = nullptr;  dphi        = nullptr;  g_res       = nullptr;  
  J0phi       = nullptr;  J0_Apar     = nullptr;  dphi_dy     = nullptr;

  if (grids_ -> Nl < 2) {
    printf("\n");
    printf("Cannot do a nonlinear run with nlaguerre < 2\n");
    printf("\n");
    exit(1);
  }

  laguerre =        new LaguerreTransform(grids_, 1);
  int nR = grids_->NxNyNz;
  red = new Block_Reduce(nR); cudaDeviceSynchronize();
  
  nBatch = grids_->Nz*grids_->Nl; 
  grad_perp_G =     new GradPerp(grids_, nBatch, grids_->NxNycNz*grids_->Nl); 
  
  nBatch = grids_->Nz*grids_->Nj; 
  grad_perp_J0phi = new GradPerp(grids_, nBatch, grids_->NxNycNz*grids_->Nj); 

  nBatch = grids_->Nz;
  grad_perp_phi =   new GradPerp(grids_, nBatch, grids_->NxNycNz);
  
  checkCuda(cudaMalloc(&dG,    sizeof(float)*grids_->NxNyNz*grids_->Nl));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMalloc(&J0phi,      sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dx,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dy,  sizeof(float)*grids_->NxNyNz*grids_->Nj));

  if (pars_->beta > 0.) {
    checkCuda(cudaMalloc(&J0_Apar,      sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
    checkCuda(cudaMalloc(&dJ0_Apar_dx,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
    checkCuda(cudaMalloc(&dJ0_Apar_dy,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
  }

  checkCuda(cudaMalloc(&dphi,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&g_res, sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  int nxyz = grids_->NxNyNz;
  int nlag = grids_->Nj;

  int nbx = min(32, nxyz);  int ngx = 1 + (nxyz-1)/nbx; 
  int nby = min(16, nlag);  int ngy = 1 + (nlag-1)/nby;

  dBx = dim3(nbx, nby, 1);
  dGx = dim3(ngx, ngy, 1);

  int nxkyz = grids_->NxNycNz;
  
  nbx = min(32, nxkyz);      ngx = 1 + (nxkyz-1)/nbx;
  nby = min(16, nlag);       ngy = 1 + (nlag-1)/nby;

  dBk = dim3(nbx, nby, 1);
  dGk = dim3(ngx, ngy, 1);

  cfl_x_inv = (float) grids_->Nx / (pars_->cfl * 2 * M_PI * pars_->x0);
  cfl_y_inv = (float) grids_->Ny / (pars_->cfl * 2 * M_PI * pars_->y0); 
  
  dt_cfl = 0.;
}

Nonlinear_GK::~Nonlinear_GK() 
{
  if ( grad_perp_G     ) delete grad_perp_G;
  if ( red             ) delete red;
  if ( laguerre        ) delete laguerre;
  if ( grad_perp_J0phi ) delete grad_perp_J0phi;
  if ( grad_perp_phi   ) delete grad_perp_phi;

  if ( dG          ) cudaFree ( dG          );
  if ( dg_dx       ) cudaFree ( dg_dx       );
  if ( dg_dy       ) cudaFree ( dg_dy       );
  if ( val1        ) cudaFree ( val1        ); 
  if ( Gy          ) cudaFree ( Gy          );
  if ( dJ0phi_dx   ) cudaFree ( dJ0phi_dx   );
  if ( dJ0phi_dy   ) cudaFree ( dJ0phi_dy   );
  if ( dJ0_Apar_dx ) cudaFree ( dJ0_Apar_dx );
  if ( dJ0_Apar_dy ) cudaFree ( dJ0_Apar_dy );
  if ( dphi        ) cudaFree ( dphi        );
  if ( g_res       ) cudaFree ( g_res       );
  if ( J0phi       ) cudaFree ( J0phi       );
  if ( J0_Apar     ) cudaFree ( J0_Apar     );
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
  
  for(int s=0; s<grids_->Nspecies; s++) {

    // BD  J0phiToGrid does not use a Laguerre transform. Implications?
    // BD  If we use alternate forms for <J0> then that would need to be reflected here

    //    printf("\n");
    //    printf("Phi:\n");
    //    qvar(f->phi, grids_->NxNycNz);

    float rho2s = pars_->species_h[s].rho2;    
    J0phiToGrid GBK (J0phi, f->phi, geo_->kperp2, laguerre->get_roots(), rho2s);

    // G->getH(J0phi); // Now G holds H
    
    grad_perp_J0phi -> dxC2R(J0phi, dJ0phi_dx);
    grad_perp_J0phi -> dyC2R(J0phi, dJ0phi_dy);

    // electromagnetic terms will couple different Hermites together. Accumulate bracket results.
    if (pars_->beta > 0.) {

      J0phiToGrid GBK (J0_Apar, f->apar, geo_->kperp2, laguerre->get_roots(), rho2s);
      
      grad_perp_J0phi -> dxC2R(J0_Apar, dJ0_Apar_dx);
      grad_perp_J0phi -> dyC2R(J0_Apar, dJ0_Apar_dy);
    }
    
    // loop over m to save memory. also makes it easier to parallelize later...
    // no extra computation: just no batching in m in FFTs and in the matrix multiplies

    // Handle m=0 separately because it will be different electromagnetically

    // ES: [ <phi>, g ] for each m -- there is no mixing of m's.
    // v_par <A> projects onto every equation. The first couple are special
    // EM, the bracket is [ (<phi> - v_par <A>), h ]
    //                  = [ (<phi> - v_par <A>), g + <phi> ]
    //                  = [ <phi>, g ] - v_par [ <A>, g ] - v_par [ <A>, <phi> ]
    //
    // With respect to the Hermite polynomials, the last term projects only onto the m=1 time evolution equation.
    //   The contribution is - v_t [<A>, <phi>], to the m=1 equation set only (ie, for all l values)
    // 
    // We already have the first term.
    //
    // The middle term contributes -v_t [ <A>, G(m=1) ] to the m=0 time evolution equation
    // The middle term contributes -v_t ( sqrt(m+1) [ <A>, G(m+1) ] + sqrt(m) [ <A>, G(m-1) ] ) to the mth equation
    // The middle term contributes -v_t sqrt(Nm-1) [ <A>, G(Nm-2) ] to the last m equation (m = Nm-1).
    //
    // So the m=0 equation should have [ <phi>, G(m=0) ] - v_t [ <A>, G(m=1) ] 
    //
    //////////////////////////////////////////////////////////////////////////////////
    //
    // And the m=1 equation should have
    //  [ <phi>, G(m=1) ] - v_t [ sqrt(2) <A>, G(m=2) ] - v_t [ <A>, G(m=0) ] - v_t [ <A>, <phi> ]
    //
    //////////////////////////////////////////////////////////////////////////////////
    // 
    //  [ <phi>, H(m=0) ] - v_t [ <A>, H(m=1) ]               [[[  m = 0 time evol. eqn ]]]
    // 
    //////////////////////////////////////////////////////////////////////////////////
    //
    //  [ <phi>, H(m=1) ] - v_t [ sqrt(2) <A>, H(m=2) ] - v_t [ <A>, H(m=0) ]      [[[ m = 1 time evol. eqn ]]]
    //
    //////////////////////////////////////////////////////////////////////////////////
    //
    //  [ <phi>, H(m=M) ] - v_t [ sqrt(M+1) <A>, H(M+1) ] - v_t sqrt(M) [ <A>, H(M-1) ]  [[[ m = M time evol. eqn ]]]
    //
    ///////////////////////////////////////////////////////////////////////////////////
    // 
    //  [ <phi>, H(m=Nm-1) ] - v_t sqrt(Nm-1) [ <A>, H(Nm-2) ]  [[[ m = Nm-1 time evol. eqn ]]]
    // 
    //////////////////////////////////////////////////////////////////////////////////
    //
    // Note that H = G + <phi> for m=0, and H = G otherwise. So we can write the whole bracket 
    // directly as [ <chi>, H ] for convenience, just as we deal with other parts of the system.
    //
    ////////////////////////////////////////////
    //
    // m=0: 
    // calculate [ <phi>, H(0) ] and [ <A>, H(1) ] .... and [ <A>, H(0) ] , too?
    //
    // m=1: 
    // calculate [ <phi>, H(1) ] and [ <A>, H(2) ] and [ <A>, H(0) ] if not saved
    //
    // m=2:
    // calculate [ <phi>, H(2) ] and [ <A>, H(3) ] and [ <A>, H(1) ] if not saved
    //
    // m=3:
    // calculate [ <phi>, H(3) ] and [ <A>, H(4) ] and [ <A>, H(2) ] if not saved


        ////////////////////////////////////////////

    // Before starting the m loops:
    // calculate <phi>, <A>, all their d/dx, d/dy derivatives
    // call them dphi/dx, dphi/dy, dA/dx, dA/dy
    // get phi, A
    //
    // get H = H(0), H_x, H_y, etc. 
    // put (phi, H) into M=0
    // put (-A,  H) into M=1
    //
    // get H = H(1), H_x, H_y, etc.
    // put (-A,  H) into M=0  (with multiplier)
    // put (phi, H) into M=1
    // put (-A,  H) into M=2  (with multiplier)
    //
    // H = H(2)
    // put (-A,  H) into M=1  (with multiplier)
    // put (phi, H) into M=2
    // put (-A,  H) into M=3  (with multiplier)
    //
    // H = H(3)
    // put (-A,  H) into M=2  (with multiplier)
    // put (phi, H) into M=3
    // put (-A,  H) into M=4  (with multiplier)
    // 
    //...
    // 
    // H = H(Nm-2)
    // put (-A,  H) into M=Nm-3   (with multiplier)
    // put (phi, H) into M=Nm-2
    // put (-A,  H) into M=Nm-1   (with multiplier)
    //
    // H = H(Nm-1)
    // put (-A,  H) into M=Nm-2   (with multiplier)
    // put (phi, H) into M=Nm-1
    

    int m=0;
    grad_perp_G -> dxC2R(G->Gm(m,s), dG);
    laguerre    -> transformToGrid(dG, dg_dx);
    
    grad_perp_G -> dyC2R(G->Gm(m,s), dG);      
    laguerre    -> transformToGrid(dG, dg_dy);
    
    bracket GBX (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);

    laguerre->transformToSpectral(g_res, dG);
    grad_perp_G->R2C(dG, G_res->Gm(m,s));
      
    for(int m=1; m<grids_->Nm-1; m++) {
      
      grad_perp_G -> dxC2R(G->Gm(m,s), dG);
      laguerre    -> transformToGrid(dG, dg_dx);
    
      grad_perp_G -> dyC2R(G->Gm(m,s), dG);      
      laguerre    -> transformToGrid(dG, dg_dy);
         
      bracket GBX (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);

      laguerre->transformToSpectral(g_res, dG);
      grad_perp_G->R2C(dG, G_res->Gm(m,s));
    }

    // Handle m=Nm-1 separately because it will be different electromagnetically
    if (grids_->Nm > 1) {

      int m=grids_->Nm-1;
      grad_perp_G -> dxC2R(G->Gm(m,s), dG);
      laguerre    -> transformToGrid(dG, dg_dx);
      
      grad_perp_G -> dyC2R(G->Gm(m,s), dG);      
      laguerre    -> transformToGrid(dG, dg_dy);
      
      bracket GBX (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);
      
      laguerre->transformToSpectral(g_res, dG);
      grad_perp_G->R2C(dG, G_res->Gm(m,s));
    }
    //    G->getG(J0phi); // now G is back to being G
  }
}
double Nonlinear_GK::cfl(Fields *f, double dt_max)
{

  grad_perp_phi -> dxC2R(f->phi, dphi);  red->Max(dphi, val1); CP_TO_CPU(vmax_y, val1, sizeof(float));
  grad_perp_phi -> dyC2R(f->phi, dphi);  red->Max(dphi, val1); CP_TO_CPU(vmax_x, val1, sizeof(float));
  // need em evaluation if beta > 0
  float vmax = max(vmax_x[0]*cfl_x_inv, vmax_y[0]*cfl_y_inv);
  dt_cfl = min(dt_max, 1./vmax);
  return dt_cfl;

}

//==============================================
// Nonlinear_KREHM
// object for handling non-linear terms in KREHM
//==============================================
Nonlinear_KREHM::Nonlinear_KREHM(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), red(nullptr), grad_perp(nullptr)
{
  tmp_c = nullptr;
  tmp_r = nullptr;
  dg_dx = nullptr;
  dg_dy = nullptr;
  dphi_dx = nullptr;
  dphi_dy = nullptr;
  dapar_dx = nullptr;
  dapar_dy = nullptr;
  
  nBatch = grids_->Nz; 
  grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz); 

  int nR = grids_->NxNyNz;
  red = new Block_Reduce(nR); cudaDeviceSynchronize();
  
  checkCuda(cudaMalloc(&tmp_c, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc(&tmp_r, sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->NxNyNz));

  checkCuda(cudaMalloc(&dphi_dx,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dphi_dy,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dapar_dx,  sizeof(float)*grids_->NxNyNz));
  checkCuda(cudaMalloc(&dapar_dy,  sizeof(float)*grids_->NxNyNz));

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  int nxyz = grids_->NxNyNz;

  int nbx = min(32, nxyz);  int ngx = 1 + (nxyz-1)/nbx; 

  dBx = dim3(nbx, 1, 1);
  dGx = dim3(ngx, 1, 1);

  int nxkyz = grids_->NxNycNz;
  
  nbx = min(32, nxkyz);      ngx = 1 + (nxkyz-1)/nbx;

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
  if ( grad_perp ) delete grad_perp;
  if ( dg_dx ) cudaFree ( dg_dx );
  if ( dg_dy ) cudaFree ( dg_dy );
  if ( tmp_r ) cudaFree ( tmp_r );
  if ( tmp_c ) cudaFree ( tmp_c );
  if ( dphi_dx ) cudaFree ( dphi_dx );
  if ( dphi_dy ) cudaFree ( dphi_dy );
  if ( dapar_dx ) cudaFree ( dapar_dx );
  if ( dapar_dy ) cudaFree ( dapar_dy );
  if ( val1 ) cudaFree ( val1 ); 
  if ( red ) delete red;
}

void Nonlinear_KREHM::nlps(MomentsG* G, Fields* f, MomentsG* G_nl)
{
  grad_perp->dxC2R(f->phi, dphi_dx);
  grad_perp->dyC2R(f->phi, dphi_dy);
  grad_perp->dxC2R(f->apar, dapar_dx);
  grad_perp->dyC2R(f->apar, dapar_dy);

  G_nl->set_zero();

  // loop over m, computing all nonlinear terms involving a bracket of g_m
  // this way each g_m only needs to be transformed twice (for d/dx and d/dy)
  for(int m=0; m<grids_->Nm-1; m++) {
    grad_perp->dxC2R(G->Gm(m), dg_dx);
    grad_perp->dyC2R(G->Gm(m), dg_dy);      

    // compute {g_m, phi}
    bracket GBX (tmp_r, dg_dx, dphi_dy, dg_dy, dphi_dx, 1.);
    grad_perp->R2C(tmp_r, tmp_c);
    // NL_m += {g_m, phi}
    add_scaled_singlemom_kernel GBK (G_nl->Gm(m), 1., G_nl->Gm(m), 1., tmp_c);

    // compute {g_m, Apar}
    bracket GBX (tmp_r, dg_dx, dapar_dy, dg_dy, dapar_dx, 1.);
    grad_perp->R2C(tmp_r, tmp_c);
    // NL_{m+1} += -rho_s/d_e*sqrt(m+1)*{g_m, Apar}
    if(m+1 < grids_->Nm-1) add_scaled_singlemom_kernel GBK (G_nl->Gm(m+1), 1., G_nl->Gm(m+1), -rho_s/d_e*sqrtf(m+1), tmp_c);
    // NL_{m-1} += -rho_s/d_e*sqrt(m)*{g_m, Apar}
    if(m>0) add_scaled_singlemom_kernel GBK (G_nl->Gm(m-1), 1., G_nl->Gm(m-1), -rho_s/d_e*sqrtf(m), tmp_c);
  }
}
double Nonlinear_KREHM::cfl(Fields *f, double dt_max)
{
  red->Max(dphi_dx, val1); CP_TO_CPU(vPhi_max_y, val1, sizeof(float));
  red->Max(dphi_dy, val1); CP_TO_CPU(vPhi_max_x, val1, sizeof(float));
  red->Max(dapar_dx, val1); CP_TO_CPU(vA_max_y, val1, sizeof(float));
  red->Max(dapar_dy, val1); CP_TO_CPU(vA_max_x, val1, sizeof(float));

  float vPhi_max = max(vPhi_max_x[0]*cfl_x_inv, vPhi_max_y[0]*cfl_y_inv);
  float vA_max = max(vA_max_x[0]*cfl_x_inv, vA_max_y[0]*cfl_y_inv);
  float vmax = vPhi_max; // add vA condition
  dt_cfl = min(dt_max, 1./vmax);
  return dt_cfl;

}

//===========================================
// Nonlinear_KS
// object for handling non-linear terms in KS
//===========================================
Nonlinear_KS::Nonlinear_KS(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), grad_perp_G(nullptr)
{

  Gy          = nullptr;
  dg_dy       = nullptr;
  g_res       = nullptr;  
  
  nBatch = 1;
  grad_perp_G =     new GradPerp(grids_, nBatch, grids_->Nyc);
  
  checkCuda(cudaMalloc(&Gy,    sizeof(float)*grids_->Ny));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->Ny));  
  checkCuda(cudaMalloc(&g_res, sizeof(float)*grids_->Ny));
  
  int nbx = min(128, grids_->Ny);
  int ngx = 1 + (grids_->Ny-1)/nbx;
  dBx = dim3(nbx, 1, 1);
  dGx = dim3(ngx, 1, 1);
  
}

Nonlinear_KS::~Nonlinear_KS() 
{
  if ( grad_perp_G     ) delete grad_perp_G;

  if ( Gy          ) cudaFree ( Gy          );  
  if ( dg_dy       ) cudaFree ( dg_dy       );
  if ( g_res       ) cudaFree ( g_res       );
}

void Nonlinear_KS::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void Nonlinear_KS::qvar (float* G, int N)
{
  float* G_h;
  int N_ = grids_->Ny;  G_h = (float*) malloc (sizeof(float)*N);

  for (int i=0; i<N; i++) {G_h[i] = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = %e \n", i%N_, i/N_, G_h[i]);
  printf("\n");

  free (G_h);
}

void Nonlinear_KS::nlps(MomentsG* G, Fields* f, MomentsG* G_res)
{

  grad_perp_G -> dyC2R(G->G(), dg_dy);
  grad_perp_G -> C2R(G->G(), Gy);
  nlks GBX (g_res, Gy, dg_dy);
  grad_perp_G -> R2C(g_res, G_res->G());
  
}
double Nonlinear_KS::cfl(Fields *f, double dt_max)
{
  return dt_max;
}

//===========================================
// Nonlinear_VP
// object for handling non-linear terms in VP
//===========================================
Nonlinear_VP::Nonlinear_VP(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), grad_perp_G(nullptr), grad_perp_phi(nullptr)
{

  Gy          = nullptr;  dphi_dy     = nullptr;  g_res       = nullptr;  
  
  nBatch = grids_->Nm;
  grad_perp_G =    new GradPerp(grids_, nBatch, grids_->Nyc*grids_->Nm);
  
  nBatch = 1;
  grad_perp_phi =  new GradPerp(grids_, nBatch, grids_->Nyc);
  
  checkCuda(cudaMalloc(&Gy,      sizeof(float)*grids_->Ny*grids_->Nm)); 
  checkCuda(cudaMalloc(&dphi_dy, sizeof(float)*grids_->Ny));              
  checkCuda(cudaMalloc(&g_res,   sizeof(float)*grids_->Ny*grids_->Nm));
  
  int nnx = grids_->Ny;    int nbx = min(32, nnx);    int ngx = 1 + (nnx-1)/nbx;
  int nny = grids_->Nm;    int nby = min(32, nny);    int ngy = 1 + (nny-1)/nby;
  
  dBx = dim3(nbx, nby, 1);
  dGx = dim3(ngx, ngy, 1);
  
}

Nonlinear_VP::~Nonlinear_VP() 
{
  if ( grad_perp_G     ) delete grad_perp_G;
  if ( grad_perp_phi   ) delete grad_perp_phi;

  if ( Gy          ) cudaFree ( Gy          );
  if ( dphi_dy     ) cudaFree ( dphi_dy     );
  if ( g_res       ) cudaFree ( g_res       );
}

void Nonlinear_VP::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void Nonlinear_VP::qvar (float* G, int N)
{
  float* G_h;
  int N_ = grids_->Ny;  G_h = (float*) malloc (sizeof(float)*N);

  for (int i=0; i<N; i++) {G_h[i] = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = %e \n", i%N_, i/N_, G_h[i]);
  printf("\n");

  free (G_h);
}

void Nonlinear_VP::nlps(MomentsG* G, Fields* f, MomentsG* G_res)
{

  grad_perp_G -> C2R(G->G(), Gy);
  grad_perp_phi -> dyC2R(f->phi, dphi_dy);
  nlvp GBX (g_res, Gy, dphi_dy);
  grad_perp_G -> R2C(g_res, G_res->G());
}

double Nonlinear_VP::cfl(Fields *f, double dt_max)
{
  return dt_max;
}
