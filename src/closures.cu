#include "closures.h"
#define GB <<< dimGrid, dimBlock >>>

Beer42::Beer42(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par_in): 
  pars_(pars), grids_(grids), grad_par_(grad_par_in), omegad_(geo->omegad), gpar_(geo->gradpar), tmp(nullptr), nu(nullptr)
{
  cudaMalloc ((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);

  D_par    = 2.*sqrt(M_PI)/(3.0*M_PI-8.0);
  Beta_par = (32.-9.*M_PI)/(3.*M_PI-8.);
  D_perp   =    sqrt(M_PI)/2.;

  cuComplex nu_h[11];
  nu_h[0].x  =  0.;    // dummy
  nu_h[0].y  =  0.;    // dummy
  nu_h[1].x  =  2.019;
  nu_h[1].y  = -1.620;
  nu_h[2].x  =  0.433;  
  nu_h[2].y  =  1.018;
  nu_h[3].x  = -0.256; 
  nu_h[3].y  =  1.487; 
  nu_h[4].x  = -0.070; 
  nu_h[4].y  = -1.382;
  nu_h[5].x  = -8.927;
  nu_h[5].y  = 12.649;
  nu_h[6].x  =  8.094;
  nu_h[6].y  = 12.638 - 11.;
  nu_h[7].x  = 13.720;
  nu_h[7].y  =  5.139 - 3.;
  nu_h[8].x  =  3.368; 
  nu_h[8].y  = -8.110;
  nu_h[9].x  =  1.974; 
  nu_h[9].y  = -1.984 - 1.;
  nu_h[10].x =  8.269;
  nu_h[10].y =  2.060 - 7.;

  cudaMalloc ((void**) &nu, sizeof(cuComplex)*11);  CP_TO_GPU (nu, nu_h, sizeof(cuComplex)*11);

  // 1d thread blocks over xyz
  int nxyz = grids_->NxNycNz;
  dimBlock = 512;
  dimGrid = (nxyz-1)/dimBlock.x + 1;
}

Beer42::~Beer42() {
  if (tmp) cudaFree(tmp);
  if (nu)  cudaFree(nu);
}

void Beer42::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
  const float vt_ = G->species->vt;

  int nn = grids_->NxNycNz; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;
  cuComplex zero = make_cuComplex(0.,0.);

  // mask unevolved moments
  setval <<< nb, nt >>> (GRhs->G(1, 2), zero, nn);
  setval <<< nb, nt >>> (GRhs->G(1, 3), zero, nn);

  //    cudaMemset(GRhs->G(1, 2, is), 0., sizeof(cuComplex)*grids_->NxNycNz);
  //    cudaMemset(GRhs->G(1, 3, is), 0., sizeof(cuComplex)*grids_->NxNycNz);

  // parallel streaming
  grad_par_->dz(G->G(0, 2), tmp); // roughly d/dz T_par
  
  add_scaled_singlemom_kernel GB (GRhs->G(0, 3), 1., GRhs->G(0, 3), -Beta_par/sqrt(3.)*gpar_*vt_, tmp);

  grad_par_->abs_dz(G->G(0, 3), tmp); // roughly |d/dz| q_par,par

  add_scaled_singlemom_kernel GB (GRhs->G(0, 3), 1., GRhs->G(0, 3), -sqrt(2.)*D_par*gpar_*vt_, tmp);

  grad_par_->abs_dz(G->G(1, 1), tmp); // rougly |d/dz| q_par,perp

  add_scaled_singlemom_kernel GB (GRhs->G(1, 1), 1., GRhs->G(1, 1), -sqrt(2.)*D_perp*gpar_*vt_, tmp);
  
  
  // toroidal terms.  
  //  beer_toroidal_closures GB (G->G(), GRhs->G(), omegad_, nu, pars_->species);
  beer_toroidal_closures GB (G->G(), GRhs->G(), omegad_, nu, G->species->tz);
}

SmithPerp::SmithPerp(Parameters* pars, Grids* grids, Geometry* geo): 
  pars_(pars), grids_(grids), omegad_(geo->omegad), Aclos_(nullptr)
{
  int q_ = pars_->smith_perp_q;
  cuComplex Aclos_h[q_];

  // hard code these cases for now...
  if(grids_->Nl==4 && q_==3) {
    Aclos_h[0].x = -2.10807;
    Aclos_h[0].y =  0.574549;
    Aclos_h[1].x = -1.25931;
    Aclos_h[1].y =  0.80951;
    Aclos_h[2].x = -0.181713;
    Aclos_h[2].y =  0.249684;
  }
  else if(grids_->Nl==5 && q_==3) {
    Aclos_h[0].x = -2.24233;
    Aclos_h[0].y =  0.551885;
    Aclos_h[1].x = -1.49324;
    Aclos_h[1].y =  0.836156;
    Aclos_h[2].x = -0.272805;
    Aclos_h[2].y =  0.292545;
  }
  else if(grids_->Nl==5 && q_==4) {
    Aclos_h[0].x = -2.8197;
    Aclos_h[0].y =  0.679165;
    Aclos_h[1].x = -2.63724;
    Aclos_h[1].y =  1.4362;
    Aclos_h[2].x = -0.896854;
    Aclos_h[2].y =  0.920214;
    Aclos_h[3].x = -0.0731348;
    Aclos_h[3].y =  0.167287;
  }
  else if(grids_->Nl==6 && q_==3) {
    Aclos_h[0].x = -2.33763;
    Aclos_h[0].y =  0.527272;
    Aclos_h[1].x = -1.66731;
    Aclos_h[1].y =  0.835661;
    Aclos_h[2].x = -0.346277;
    Aclos_h[2].y =  0.313484;
  }
  else if(grids_->Nl==6 && q_==4) {
    Aclos_h[0].x = -2.97138;
    Aclos_h[0].y =  0.66065;
    Aclos_h[1].x = -3.01477;
    Aclos_h[1].y =  1.48449;
    Aclos_h[2].x = -1.18109;
    Aclos_h[2].y =  1.04256;
    Aclos_h[3].x = -0.13387;
    Aclos_h[3].y =  0.221786;
  }
  else if(grids_->Nl==6 && q_==5) {
    Aclos_h[0].x = -3.53482;
    Aclos_h[0].y =  0.771299;
    Aclos_h[1].x = -4.52836;
    Aclos_h[1].y =  2.17916;
    Aclos_h[2].x = -2.50811;
    Aclos_h[2].y =  2.14358;
    Aclos_h[3].x = -0.537375;
    Aclos_h[3].y =  0.839509;
    Aclos_h[4].x = -0.0226971;
    Aclos_h[4].y =  0.102349;
  }
  else if(grids_->Nl==8 && q_==4) {
    Aclos_h[0].x = -3.17353;
    Aclos_h[0].y =  0.616513;
    Aclos_h[1].x = -3.54865;
    Aclos_h[1].y =  1.48678;
    Aclos_h[2].x = -1.62127;
    Aclos_h[2].y =  1.15187;
    Aclos_h[3].x = -0.244346;
    Aclos_h[3].y =  0.283442;
  } else if(grids_->Nl==8 && q_==3) {
    Aclos_h[0].x = -2.46437;
    Aclos_h[0].y =  0.482544;
    Aclos_h[1].x = -1.90784;
    Aclos_h[1].y =  0.806717;
    Aclos_h[2].x = -0.454126;
    Aclos_h[2].y =  0.32645;
  }
  else {
    printf("ERROR: specified Smith closure not yet implemented\n");
    exit(1);
  }

  cudaMalloc ((void**) &Aclos_, sizeof(cuComplex)*q_);  CP_TO_GPU (Aclos_, Aclos_h, sizeof(cuComplex)*q_);

  // 1d thread blocks over xyz
  int nxyz = grids_->NxNycNz;
  dimBlock = 512;
  dimGrid = (nxyz-1)/dimBlock.x + 1;
}

SmithPerp::~SmithPerp() {
  if (Aclos_) cudaFree(Aclos_);
}

void SmithPerp::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
  // perp closure terms are only toroidal
  smith_perp_toroidal_closures GB (G->G(), GRhs->G(), omegad_, Aclos_, q_, G->species->tz);
}


SmithPar::SmithPar(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par_in):  
  pars_(pars), grids_(grids), grad_par_(grad_par_in), gpar_(geo->gradpar),
  tmp(nullptr), tmp_abs(nullptr), clos(nullptr), a_coefficients_(nullptr)
{ 

  int q_ = pars_->smith_par_q;
  cudaMalloc ((void**) &tmp,     sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc ((void**) &tmp_abs, sizeof(cuComplex)*grids_->NxNycNz);
  
  // allocate closure array
  cudaMalloc ((void**) &clos,    sizeof(cuComplex)*grids_->NxNycNz);

  // calculate closure coefficients 
  a_coefficients_ = (cuComplex*) malloc (q_*sizeof(cuComplex));
  smith_par_getAs (grids->Nm, q_, a_coefficients_);

  // 1d thread blocks over xyz
  int nxyz = grids_->NxNycNz;
  dimBlock = 512;
  dimGrid = (nxyz-1)/dimBlock.x + 1;

}

SmithPar::~SmithPar() {
  if (a_coefficients_) free(a_coefficients_);

  if (clos)        cudaFree(clos);
  if (tmp)         cudaFree(tmp);
  if (tmp_abs)     cudaFree(tmp_abs);
}

void SmithPar::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
  int M = grids_->Nm - 1;

  const float vt_ = G->species->vt;

  // apply closure to mth hermite equation for all laguerre moments
  for(int l = 0; l < grids_->Nl; l++) {
    
    //      cudaMemset(clos, 0, sizeof(cuComplex)*grids_->NxNycNz);
    
    int nn = grids_->NxNycNz; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;
    cuComplex zero = make_cuComplex(0.,0.);
    
    // reset closure array every time step
    setval <<< nb, nt >>> (clos, zero, nn); 

  // write m+1 moment as a sum of lower order moments
    for (int m = M; m >= grids_->Nm - q_; m--) {

      grad_par_->dz(G->G(l, m), tmp);              // Aha. This is where the difference between kzp and kz comes in
      grad_par_->abs_dz(G->G(l, m), tmp_abs);
      
      add_scaled_singlemom_kernel GB (clos, 1., clos, -a_coefficients_[M - m].y, tmp_abs, a_coefficients_[M - m].x, tmp);
    }
    add_scaled_singlemom_kernel GB (GRhs->G(l, M), 1., GRhs->G(l, M), -sqrtf(M+1)*gpar_*vt_, clos);
  }
}
