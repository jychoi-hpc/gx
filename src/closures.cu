#include "closures.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "smith_par_closure.h"

Beer42::Beer42(Grids* grids, const Geometry* geo, GradParallel* grad_par_in): 
    grids_(grids), grad_par(grad_par_in), omegad_(geo->omegad), gpar_(geo->gradpar)
{
  cudaMalloc((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);

  D_par = 2.*sqrt(M_PI)/(3.0*M_PI-8.0);
  D_perp = sqrt(M_PI)/2.;
  Beta_par = (32.-9.*M_PI)/(3.*M_PI-8.);

  cuComplex nu_h[11];
  nu_h[0].x = 0.; // dummy
  nu_h[0].y = 0.; // dummy
  nu_h[1].x=2.019;
  nu_h[1].y=-1.620;
  nu_h[2].x=.433;  
  nu_h[2].y= 1.018;
  nu_h[3].x=-.256; 
  nu_h[3].y=1.487; 
  nu_h[4].x=-.070; 
  nu_h[4].y=-1.382;
  nu_h[5].x=-8.927;
  nu_h[5].y=12.649;
  nu_h[6].x= 8.094;
  nu_h[6].y= 12.638-11.;
  nu_h[7].x= 13.720;
  nu_h[7].y= 5.139-3.;
  nu_h[8].x= 3.368; 
  nu_h[8].y= -8.110;
  nu_h[9].x= 1.974; 
  nu_h[9].y= -1.984-1.;
  nu_h[10].x= 8.269;
  nu_h[10].y= 2.060-7.;

  cudaMalloc((void**) &nu, sizeof(cuComplex)*11);
  CP_TO_GPU(nu, nu_h, sizeof(cuComplex)*11);

  // 1d thread blocks over xyz
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
}

Beer42::~Beer42() {
  cudaFree(tmp);
  cudaFree(nu);
}

int Beer42::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
  // mask unevolved moments
  cudaMemset(GRhs->G(1,2), 0., sizeof(cuComplex)*grids_->NxNycNz);
  cudaMemset(GRhs->G(1,3), 0., sizeof(cuComplex)*grids_->NxNycNz);

  // parallel terms (each must be done separately because of FFTs)
  grad_par->dz(G->G(0,2), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (GRhs->G(0,3), 1., GRhs->G(0,3), -Beta_par/sqrt(3.)*gpar_, tmp);

  grad_par->abs_dz(G->G(0,3), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (GRhs->G(0,3), 1., GRhs->G(0,3), -sqrt(2.)*D_par*gpar_, tmp);

  grad_par->abs_dz(G->G(1,1), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (GRhs->G(1,1), 1., GRhs->G(1,1), -sqrt(2.)*D_perp*gpar_, tmp);
  
  // toroidal terms
  beer_toroidal_closures<<<dimGrid,dimBlock>>>(G->G(), GRhs->G(), omegad_, nu);

  return 0;
}


SmithPerp::SmithPerp(Grids* grids, const Geometry* geo, int q, cuComplex w0): 
    grids_(grids), omegad_(geo->omegad), q_(q)
{
  cuComplex Aclos_h[q_];

  // hard code these cases for now...
  if(grids_->Nl==4 && q_==3) {
    Aclos_h[0].x = -2.10807;
    Aclos_h[0].y = 0.574549;
    Aclos_h[1].x = -1.25931;
    Aclos_h[1].y = 0.80951;
    Aclos_h[2].x = -0.181713;
    Aclos_h[2].y = 0.249684;
  }
  else if(grids_->Nl==5 && q_==3) {
    Aclos_h[0].x = -2.24233;
    Aclos_h[0].y = 0.551885;
    Aclos_h[1].x = -1.49324;
    Aclos_h[1].y = 0.836156;
    Aclos_h[2].x = -0.272805;
    Aclos_h[2].y = 0.292545;
  }
  else if(grids_->Nl==5 && q_==4) {
    Aclos_h[0].x = -2.8197;
    Aclos_h[0].y = 0.679165;
    Aclos_h[1].x = -2.63724;
    Aclos_h[1].y = 1.4362;
    Aclos_h[2].x = -0.896854;
    Aclos_h[2].y = 0.920214;
    Aclos_h[3].x = -0.0731348;
    Aclos_h[3].y = 0.167287;
  }
  else if(grids_->Nl==6 && q_==3) {
    Aclos_h[0].x = -2.33763;
    Aclos_h[0].y = 0.527272;
    Aclos_h[1].x = -1.66731;
    Aclos_h[1].y = 0.835661;
    Aclos_h[2].x = -0.346277;
    Aclos_h[2].y = 0.313484;
  }
  else if(grids_->Nl==6 && q_==4) {
    Aclos_h[0].x = -2.97138;
    Aclos_h[0].y = 0.66065;
    Aclos_h[1].x = -3.01477;
    Aclos_h[1].y = 1.48449;
    Aclos_h[2].x = -1.18109;
    Aclos_h[2].y = 1.04256;
    Aclos_h[3].x = -0.13387;
    Aclos_h[3].y = 0.221786;
  }
  else if(grids_->Nl==6 && q_==5) {
    Aclos_h[0].x = -3.53482;
    Aclos_h[0].y = 0.771299;
    Aclos_h[1].x = -4.52836;
    Aclos_h[1].y = 2.17916;
    Aclos_h[2].x = -2.50811;
    Aclos_h[2].y = 2.14358;
    Aclos_h[3].x = -0.537375;
    Aclos_h[3].y = 0.839509;
    Aclos_h[4].x = -0.0226971;
    Aclos_h[4].y = 0.102349;
  }
  else if(grids_->Nl==8 && q_==4) {
    Aclos_h[0].x = -3.17353;
    Aclos_h[0].y = 0.616513;
    Aclos_h[1].x = -3.54865;
    Aclos_h[1].y = 1.48678;
    Aclos_h[2].x = -1.62127;
    Aclos_h[2].y = 1.15187;
    Aclos_h[3].x = -0.244346;
    Aclos_h[3].y = 0.283442;
  } else if(grids_->Nl==8 && q_==3) {
    Aclos_h[0].x = -2.46437;
    Aclos_h[0].y = 0.482544;
    Aclos_h[1].x = -1.90784;
    Aclos_h[1].y = 0.806717;
    Aclos_h[2].x = -0.454126;
    Aclos_h[2].y = 0.32645;
  }
  else {
    printf("ERROR: specified Smith closure not yet implemented\n");
    exit(1);
  }

  cudaMalloc((void**) &Aclos_, sizeof(cuComplex)*q_);
  CP_TO_GPU(Aclos_, Aclos_h, sizeof(cuComplex)*q_);

  // 1d thread blocks over xyz
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
}

SmithPerp::~SmithPerp() {
  cudaFree(Aclos_);
}

int SmithPerp::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
  // perp closure terms are only toroidal
  smith_perp_toroidal_closures<<<dimGrid,dimBlock>>>(G->G(), GRhs->G(), omegad_, Aclos_, q_);

  return 0;
}

SmithPar::SmithPar(Grids* grids, const Geometry* geo, GradParallel* grad_par_in, int q): 
    grids_(grids), grad_par(grad_par_in), gpar_(geo->gradpar), q_(q)
{ 
  cudaMalloc((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &tmp_abs, sizeof(cuComplex)*grids_->NxNycNz);
  
  // allocate closure array
  cudaMalloc((void**) &clos, grids_->NxNycNz*sizeof(cuComplex));

  // calculate closure coefficients 
  a_coefficients_ = (cuComplex*) malloc(q_*sizeof(cuComplex));
  smith_par_getAs(grids->Nm, q_, a_coefficients_);

  // 1d thread blocks over xyz
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
}

SmithPar::~SmithPar() {
  free(a_coefficients_);
  cudaFree(clos);
  cudaFree(tmp);
  cudaFree(tmp_abs);
}

int SmithPar::apply_closures(MomentsG* G, MomentsG* GRhs) 
{
    int M = grids_->Nm - 1;

    // apply closure to mth hermite equation for all laguerre moments
    for(int l = 0; l < grids_->Nl; l++) {
      
      // reset closure array every time step
      cudaMemset(clos, 0, grids_->NxNycNz*sizeof(cuComplex));

      // write m+1 moment as a sum of lower order moments
      for (int m = M; m >= grids_->Nm - q_; m--) {
          grad_par->dz(G->G(l,m), tmp);
          grad_par->abs_dz(G->G(l,m), tmp_abs);
          add_scaled_singlemom_kernel <<<dimGrid,dimBlock>>>
	    (clos, 1., clos, -a_coefficients_[M - m].y, tmp_abs, a_coefficients_[M - m].x, tmp);
      }

      add_scaled_singlemom_kernel <<<dimGrid,dimBlock>>>
	(GRhs->G(l,M), 1., GRhs->G(l,M), -sqrt(M+1)*gpar_, clos);
    }
    
    return 0;
}
