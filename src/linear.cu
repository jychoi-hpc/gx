#include "linear.h"
#include "device_funcs.h"
#include "cufft.h"
#include "get_error.h"
#include "species.h"
#include "cuda_constants.h"

Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  GRhs_par = new MomentsG(pars_, grids_);

  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  else if(pars_->boundary_option_periodic) {
    DEBUGPRINT("Using periodic for grad parallel.\n");
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    DEBUGPRINT("Using twist-and-shift for grad parallel.\n");
    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
 
  if(pars_->closure_model_opt==BEER42) {
    DEBUGPRINT("Initializing Beer 4+2 closures\n");
    closures = new Beer42(grids_, geo_, grad_par);
  } else if (pars_->closure_model_opt==SMITHPERP) {
    DEBUGPRINT("Initializing Smith perpendicular toroidal closures\n");
    closures = new SmithPerp(grids_, geo_, pars_->smith_perp_q, pars_->smith_perp_w0);
  } else if (pars_->closure_model_opt == SMITHPAR) {
    DEBUGPRINT("Initializing Smith parallel closures\n");
    closures = new SmithPar(grids_, geo_, grad_par, pars_->smith_par_q);
  }

   // allocate conservation terms for collision operator
  int size = sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);
   
  // set up CUDA grids for main linear kernel.  
  // NOTE: dimBlock.x = sharedSize.x = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more LH resolution without changing shared memory layouts.
  // dimBlock = dim3(8, min(4, grids_->Nl), min(4, grids_->Nm));
  //  dimBlock = dim3(8, min(4, grids_->Nl), min(4, grids_->Nm));
  dimBlock = dim3(pars_->i_share, min(4, grids_->Nl), min(4, grids_->Nm));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x+1, 1, 1);
  sharedSize = dimBlock.x*(grids_->Nl+2)*(grids_->Nm+4)*sizeof(cuComplex);
  DEBUGPRINT("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);
  if(sharedSize/1024.>48.) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("size of shared memory block must be less than 48 KB, so make sure (nm+4)*(nlaaguerre+2)<%d.\n", 48*1024/8/dimBlock.x);
    exit(1);
  }
}

Linear::~Linear()
{
  if(pars_->closure_model_opt>0) delete closures;
  delete grad_par;
  delete GRhs_par;
}

int Linear::rhs(MomentsG* G, Fields* f, MomentsG* GRhs) {

  // calculate conservation terms for collision operator
  conservation_terms<<<grids_->NxNycNz/256+1, 256>>>
    (upar_bar, uperp_bar, t_bar, G->G(), f->phi, geo_->kperp2, pars_->species);
  /*
  for (int is=0; is < grids_->Nspecies; is++) {
    conservation_terms<<<grids_->NxNycNz/256+1, 256>>>
      (upar_bar, uperp_bar, t_bar, G->G(), f->phi, geo_->kperp2, pars_->species[is].zt, pars_->species[is].rho2, is);
  }
  */

  // calculate RHS
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, upar_bar, uperp_bar, t_bar,
        geo_->kperp2, geo_->cv_d, geo_->gb_d, geo_->bgrad, 
       	grids_->ky, pars_->species, GRhs_par->G(), GRhs->G());

  // hypercollisions
  if(pars_->hypercollisions) {
    hypercollisions<<<dimGrid,dimBlock>>>(G->G(), pars_->nu_hyper_l, pars_->nu_hyper_m,
					  pars_->p_hyper_l, pars_->p_hyper_m, GRhs->G());
  }

  // parallel gradient term
  grad_par->dz(GRhs_par);

  // combine
  GRhs->add_scaled(1., GRhs, (float) geo_->gradpar, GRhs_par);
  
  // closures
  if(pars_->closure_model_opt>0) {
    closures->apply_closures(G, GRhs);
  }

  return 0;
}


