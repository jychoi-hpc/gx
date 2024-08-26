#include "linear.h"

//=======================================
// Linear_GK
// object for handling linear terms in GK
//=======================================
Linear_GK::Linear_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo), 
  closures(nullptr), grad_par(nullptr)
{
  ks = false;
  vp = false;
  upar_bar  = nullptr;           uperp_bar = nullptr;            t_bar     = nullptr;
  favg = nullptr;
  df   = nullptr;
  s01  = nullptr;
  s10  = nullptr;
  s11  = nullptr;
  vol_fac = nullptr;
  tmpG = nullptr;
  
  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  /* the below is commented out because the GradParallelLinked implementation with nLinks = 1 is 
     faster than GradParallelPeriodic, and gives same results. so when boundary_option_periodic is requested,
      we set jtwist = 2*nx (in Parameters::set_jtwist_x0) to give nLinks = 1 for all modes and use GradParallelLinked. */
  //else if(pars_->boundary_option_periodic && pars_->nx_in > 1) {
  //  DEBUGPRINT("Using periodic for grad parallel.\n");
  //  grad_par = new GradParallelPeriodic(grids_);
  //}
  else if(pars_->nonTwist) {
    grad_par = new GradParallelNTFT(pars_, grids_);
  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }

  switch (pars_->closure_model_opt)
    {
    case Closure::none      :
      break;
    case Closure::beer42    :
      DEBUGPRINT("Initializing Beer 4+2 closures\n");
      closures = new Beer42(pars_, grids_, geo_, grad_par);
      break;
    case Closure::smithperp :
      DEBUGPRINT("Initializing Smith perpendicular toroidal closures\n");
      closures = new SmithPerp(pars_, grids_, geo_);
      break;
    case Closure::smithpar  :
      DEBUGPRINT("Initializing Smith parallel closures\n");
      closures = new SmithPar(pars_, grids_, geo_, grad_par);
      break;
    }
  
  if (pars_->HB_hyper) {
    cudaMalloc((void**) &favg, sizeof(cuComplex)*grids_->Nx);
    cudaMalloc((void**) &df,  sizeof(cuComplex)*grids_->NxNycNz);
    cudaMalloc((void**) &s01, sizeof(float));
    cudaMalloc((void**) &s10, sizeof(float)*grids_->Nz);
    cudaMalloc((void**) &s11, sizeof(float)*grids_->Nz);
    cudaMalloc((void**) &vol_fac, sizeof(float)*grids_->Nz);
    
    volDenom = 0.;  
    float *vol_fac_h;
    vol_fac_h = (float*) malloc (sizeof(float) * grids_->Nz);
    cudaMalloc (&vol_fac, sizeof(float) * grids_->Nz);
    for (int i=0; i < grids_->Nz; i++) volDenom   += geo_->jacobian_h[i]; 
    for (int i=0; i < grids_->Nz; i++) vol_fac_h[i]  = geo_->jacobian_h[i] / volDenom;
    CP_TO_GPU(vol_fac, vol_fac_h, sizeof(float)*grids_->Nz);
    free(vol_fac_h);
  }
  
  // allocate conservation terms for collision operator
  size_t size = sizeof(cuComplex)*grids_->NxNycNz;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);

  if(pars_->hypercollisions_kz) {
    tmpG = new MomentsG (pars_, grids_);
  }

  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  
  dBs = dim3(nt1, nt2, nt3);
  dGs = dim3(nb1, nb2, nb3);

  nn1 = grids_->Nyc;             nt1 = min(nn1, 16);    nb1 = (nn1-1)/nt1 + 1;
  nn2 = grids_->Nx*grids_->Nz;   nt2 = min(nn2, 16);    nb2 = (nn2-1)/nt2 + 1;
  nn3 = grids_->Nm*grids_->Nl;   nt3 = min(nn3,  4);    nb3 = (nn3-1)/nt3 + 1;
  
  dB_all = dim3(nt1, nt2, nt3);
  dG_all = dim3(nb1, nb2, nb3);	 

  // set up CUDA grids for main linear kernel.  
  // NOTE: nt1 = sharedSize = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more LH resolution without changing shared memory layouts
  // so i_share = 8 is used by default.

  nn1 = grids_->NxNycNz;         nt1 = pars_->i_share     ;   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;                       nt2 = min(grids_->Nl, 4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;                       nt3 = min(grids_->Nm, 4 );   nb3 = 1 + (nn3-1)/nt3;

  dimBlock = dim3(nt1, nt2, nt3);
  dimGrid  = dim3(nb1, nb2, nb3);
  
  if(grids_->m_ghost == 0)
    sharedSize = nt1 * (grids_->Nl+2) * (grids_->Nm+4) * sizeof(cuComplex);
  else 
    sharedSize = nt1 * (grids_->Nl+2) * (grids_->Nm+2*grids_->m_ghost) * sizeof(cuComplex);

  int dev;
  cudaDeviceProp prop;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  maxSharedSize = prop.sharedMemPerBlockOptin;

  DEBUGPRINT("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);

  if( sharedSize > maxSharedSize && grids_->m_ghost == 0) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("If you wish to try to keep this velocity resolution, ");
    printf("you can try lowering i_share in your input file.\n");
    printf("You are using i_share = %d now, perhaps by default.\n", pars_->i_share);
    printf("The size of the shared memory block should be less than %ld KB ", maxSharedSize/1024);
    printf("which means i_share*(nhermite+4)*(nlaguerre+2) < %d. \n", 12*1024);
    printf("Presently, you have set i_share*(nhermite+4)*(nlaguerre+2) = %d. \n", pars_->i_share*(grids_->Nm+4)*(grids_->Nl+2));
    exit(1);
  }

  nn1 = grids_->NxNycNz;         nt1 = min(grids_->NxNycNz, 32) ;   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nl;              nt2 = min(grids_->Nl, 4 )      ;   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nm;              nt3 = min(grids_->Nm, 4 )      ;   nb3 = 1 + (nn3-1)/nt3;

  dimBlockh = dim3(nt1, nt2, nt3);
  dimGridh  = dim3(nb1, nb2, nb3);
  
  cudaFuncSetAttribute(rhs_linear, cudaFuncAttributeMaxDynamicSharedMemorySize, 12*1024*sizeof(cuComplex));    
}

Linear_GK::~Linear_GK()
{
  if (closures) delete closures;
  if (grad_par) delete grad_par;

  if (favg)       cudaFree(favg);
  if (df)         cudaFree(df);
  if (s10)        cudaFree(s10);
  if (s11)        cudaFree(s11);
  if (upar_bar)   cudaFree(upar_bar);
  if (uperp_bar)  cudaFree(uperp_bar);
  if (t_bar)      cudaFree(t_bar);
  if (vol_fac)    cudaFree(vol_fac);
  if (tmpG) delete tmpG;
}

void Linear_GK::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) {
  // finish Hermite ghost exchange
  cudaStreamSynchronize(G->syncStream);

  // calculate conservation terms for collision operator
  int nn1 = grids_->NxNycNz;  int nt1 = min(nn1, 256);  int nb1 = 1 + (nn1-1)/nt1;
  if (pars_->collisions && pars_->coll_conservation)  conservation_terms <<< nb1, nt1 >>>
			    (upar_bar, uperp_bar, t_bar, G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));

  // Free-streaming requires parallel FFTs, so do that first
  if(grids_->Nz>1) {
    streaming_rhs <<< dGs, dBs >>> (G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, geo_->gradpar, *(G->species), GRhs->G());
    grad_par->dz(GRhs, GRhs, false);
  }
  
  // calculate most of the RHS
  cudaFuncSetAttribute(rhs_linear, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedSize);
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, f->apar, f-> bpar, upar_bar, uperp_bar, t_bar,
        geo_->kperp2, geo_->cv_d, geo_->gb_d, geo_->bmag, geo_->bgrad, 
	grids_->ky, *(G->species), pars_->species_h[0], GRhs->G(), pars_->ei_colls,
	pars_->rhoc, pars_->g_exb, pars_->RBzeta, pars_->qsf);

  // hyper model by Hammett and Belli
  if (pars_->HB_hyper) {
    
    int nt1 = min(128, grids_->Nx);
    int nb1 = 1 + (grids_->Nx*grids_->Nyc-1)/nt1;
    
    fieldlineaverage <<< nb1, nt1 >>> (favg, df, f->phi, vol_fac);

    get_s01 <<< 1, 1 >>> (s01, favg, grids_->kx, pars_->w_osc);
    nt1 = min(128, grids_->Nz);
    nb1 = 1 + (grids_->Nz-1)/nt1;
    
    get_s1 <<< nb1, nt1 >>> (s10, s11, grids_->kx, grids_->ky, df, pars_->w_osc);
    
    HB_hyper <<< dG_all, dB_all >>> (G->G(), s01, s10, s11,
				     grids_->kx, grids_->ky, pars_->D_HB, pars_->p_HB, GRhs->G());
    
  }
  
  // closures
  switch (pars_->closure_model_opt) {
  case Closure::none : break;
  case Closure::beer42 : closures->apply_closures(G, GRhs); break;
  case Closure::smithperp : closures->apply_closures(G, GRhs); break;
  case Closure::smithpar : closures->apply_closures(G, GRhs); break;
  }

  // hypercollisions with const coefficient
  if(pars_->hypercollisions_const) hypercollisions<<<dimGridh,dimBlockh>>>(G->G(),
		  						   pars_->nu_hyper_l,
								   pars_->nu_hyper_m,
								   G->species->vt/pars_->vtmax*pars_->nu_hyper_lm/dt,
								   pars_->p_hyper_l,
								   pars_->p_hyper_m, 
								   pars_->p_hyper_lm, 
								   GRhs->G(), G->species->vt);

  // hypercollisions with coefficient propto kz
  if(pars_->hypercollisions_kz) {
    float M = (float) grids_->Nm_glob-1;
    float p = (float) pars_->p_hyper_m;
    float vt = G->species->vt;
    float nu_hyp_m = pars_->nu_hyper_m*(p + 0.5)/powf(M, p + 0.5)*2.3*vt*abs(geo_->gradpar);
    tmpG->set_zero();
    hypercollisions_kz<<<dimGridh, dimBlockh>>>(G->G(), nu_hyp_m, p, tmpG->G());
    grad_par->abs_dz(tmpG, GRhs, true);
  }

  // hyper in k-space
  if(pars_->hyper) hyperdiff <<<dimGridh,dimBlockh>>>(G->G(), grids_->kx, grids_->ky,
						      pars_->p_hyper, pars_->D_hyper, GRhs->G());

  if(pars_->hyperz) grad_par->hyperz(G, GRhs, pars_->nu_hyper_z/dt, true);
 
  // apply parallel boundary conditions. for linked BCs, this involves applying 
  // a damping operator to the RHS near the boundaries of extended domain.
  if(!pars_->boundary_option_periodic && !pars_->local_limit) grad_par->applyBCs(G, GRhs, f, geo_->kperp2, dt);
}

void Linear_GK::get_max_frequency(double *omega_max)
{
  // estimate max linear frequency from kz_max*vpar_max*vt_max + omegad_max, with omegad_max ~ 2*tz_max*(kx_max+ky_max)*vpar_max^2/R

  if (geo_->shat == 0.0) {
    omega_max[0] = pars_->tzmax*grids_->kx_max
      * (grids_->vpar_max*grids_->vpar_max*abs(geo_->cvdrift0_max) + grids_->muB_max*abs(geo_->gbdrift0_max));
  } else {
    if (pars_->nonTwist) {
      omega_max[0] = pars_->tzmax*(grids_->kx_max + grids_->m0_max / pars_->x0)
        * (grids_->vpar_max*grids_->vpar_max*abs(geo_->cvdrift0_max) + grids_->muB_max*abs(geo_->gbdrift0_max));
    } else {
      omega_max[0] = pars_->tzmax*grids_->kx_max/abs(geo_->shat)
        * (grids_->vpar_max*grids_->vpar_max*abs(geo_->cvdrift0_max) + grids_->muB_max*abs(geo_->gbdrift0_max));
    }
  }
  omega_max[1] = pars_->tzmax*grids_->ky_max*
    (grids_->vpar_max*grids_->vpar_max*geo_->cvdrift_max + grids_->muB_max*geo_->gbdrift_max);
  if(pars_->linear && pars_->etamax < 1e5) {omega_max[1] = (omega_max[1] + grids_->ky_max*
	     (1 + pars_->etamax*(grids_->vpar_max*grids_->vpar_max/2 + grids_->muB_max - 1.5)));}
  float beta = pars_->beta;
  float nte = pars_->ne*pars_->Te;
  float mime = pars_->vtmax*pars_->vtmax/pars_->vtmin/pars_->vtmin;
  float kperprho2 = grids_->kperp_min*grids_->kperp_min/geo_->bmag_max/geo_->bmag_max;
  omega_max[2] = pars_->vtmax*grids_->kz_max*abs(geo_->gradpar) * 
                 max(grids_->vpar_max, pars_->nspec_in > 1 ? 1/sqrt(beta*nte/2*mime + kperprho2): 0.);
}

//==========================================
// Linear_KREHM
// object for handling linear terms in KREHM
//==========================================
Linear_KREHM::Linear_KREHM(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  closures(nullptr), grad_par(nullptr)
{
  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  //else if(pars_->boundary_option_periodic) {
  //  DEBUGPRINT("Using periodic for grad parallel.\n");
  //  grad_par = new GradParallelPeriodic(grids_);
  //}
  else {
    DEBUGPRINT("Using twist-and-shift for grad parallel.\n");
    grad_par = new GradParallelLinked(pars_, grids_);
  }
 
  switch (pars_->closure_model_opt)
  {
    case Closure::none      :
      // OK, no closure specified, no need to print anything
      break;
    default:
      // A closure was specified, but we aren't using it, best to inform the user.
      fprintf(stderr, "Ignoring closure option. Closures not supported for KREHM");
      break;
    case Closure::smithpar  :
      DEBUGPRINT("Initializing Smith parallel closures\n");
      //closures = new SmithPar(pars_, grids_, geo_, grad_par);
      break;
  }
  
  int nn1 = grids_->Nyc;   int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;    int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz;    int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  
  dBs = dim3(nt1, nt2, nt3);
  dGs = dim3(nb1, nb2, nb3);
  
  nn1 = grids_->NxNycNz;         nt1 = min(grids_->NxNycNz, 32) ;   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;                       nt2 = 1;   nb2 = 1;
  nn3 = grids_->Nm;              nt3 = min(grids_->Nm, 4 );   nb3 = 1 + (nn3-1)/nt3;

  dimBlockh = dim3(nt1, nt2, nt3);
  dimGridh  = dim3(nb1, nb2, nb3);
  
  rho_s = pars->rho_s;
  d_e = pars->d_e;
  nu_ei = pars->nu_ei;
}

Linear_KREHM::~Linear_KREHM()
{
  if (closures) delete closures;
  if (grad_par) delete grad_par;
}

void Linear_KREHM::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) {

  if(grids_->Nz>1) {
    cudaStreamSynchronize(G->syncStream);
    rhs_linear_krehm <<< dGs, dBs >>> (G->G(), f->phi, f->apar, f->apar_ext, nu_ei, rho_s, d_e, geo_->gradpar, GRhs->G());
    grad_par->dz(GRhs, GRhs, false);
  }
  
  // closures
  switch (pars_->closure_model_opt) {
  case Closure::none : break;
  case Closure::beer42 : closures->apply_closures(G, GRhs); break;
  case Closure::smithperp : closures->apply_closures(G, GRhs); break;
  case Closure::smithpar : closures->apply_closures(G, GRhs); break;
  }

  krehm_collisions <<< dGs, dBs >>> (G->G(), f->apar, f->apar_ext, grids_->kx, grids_->ky, nu_ei, rho_s, d_e, GRhs->G());

  // hypercollisions
  if(pars_->hypercollisions_const) hypercollisions<<<dimGridh,dimBlockh>>>(G->G(),
								   0., 1./dt/pars_->nm_in, 0.,
								   1, pars_->p_hyper_m, 1, 
								   GRhs->G(), 1.);
  // hyper in k-space
  if(pars_->hyper) hyperdiff <<<dimGridh,dimBlockh>>>(G->G(), grids_->kx, grids_->ky,
						      pars_->p_hyper, pars_->D_hyper/dt, GRhs->G());

}

void Linear_KREHM::get_max_frequency(double *omega_max)
{
  // estimate max linear frequency from kz_max*vpar_max
  omega_max[0] = 0.0;
  omega_max[1] = 0.0;
  omega_max[2] = max(rho_s/d_e*grids_->vpar_max*grids_->kz_max, pars_->nm_in*nu_ei);
}

//===============================================================
// Linear_cetg
// object for handling linear terms in the collisional ETG model
//===============================================================
Linear_cetg::Linear_cetg(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo), 
  grad_par(nullptr)
{
  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  //  else if(pars_->boundary_option_periodic) {
  //    DEBUGPRINT("Using periodic for grad parallel.\n");
  //    grad_par = new GradParallelPeriodic(grids_);
  //  }
  else {
    DEBUGPRINT("Using twist-and-shift for grad parallel.\n");
    grad_par = new GradParallelLinked(pars_, grids_);
  }
 
  int nn1 = grids_->Nyc;   int nt1 = min(nn1, 8);    int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;    int nt2 = min(nn2, 8);    int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz;    int nt3 = min(nn3, 8);    int nb3 = 1 + (nn3-1)/nt3;

  dBs = dim3(nt1, nt2, nt3);
  dGs = dim3(nb1, nb2, nb3);
    
  Z_ion = pars_->ion_z; 

  float denom = 1. +  61./(sqrt(128.)*Z_ion) + 9./(2.*Z_ion*Z_ion);
  
  // Defined in Adkins, Eq (B38) 
  c1 = (217./64. + 151./(sqrt(128.)*Z_ion) + 9./(2.*Z_ion*Z_ion)) / denom ;
  c2 = 2.5 * (33./16. + 45./(sqrt(128.)*Z_ion)) / denom ;
  c3 = 25./4. * (13./4. + 45./(sqrt(128.)*Z_ion)) / denom - c2*c2/c1 ; 
  // two useful combinations
  C12 = 1. + c2/c1;
  C23 = c3/c1 + C12*C12;
    
}

Linear_cetg::~Linear_cetg()
{
  if (grad_par) delete grad_par;
}

void Linear_cetg::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) {

  cudaStreamSynchronize(G->syncStream);

  GRhs->set_zero();

  rhs_diff_cetg <<< dGs, dBs >>> (G->G(0,0), G->G(1,0), f->phi, geo_->gradpar, c1, C12, C23, GRhs->G());
  grad_par->dz2(GRhs);
  rhs_lin_cetg <<< dGs, dBs >>> (f->phi, grids_->ky, GRhs->G());
  hyper_cetg <<< dGs, dBs >>> (G->G(), grids_->kx, grids_->ky, pars_->nu_hyper, pars_->D_hyper, GRhs->G());    
}

void Linear_cetg::get_max_frequency(double *omega_max)
{

  float kymax_ = (float) grids_->Ny/3./pars_->y0;
  float kzmax_ = (float) grids_->Nz/3./pars_->z0*geo_->gradpar;
  float cfac_  = 0.5 * c1 * sqrt(1.0+c2/c1);
  
  omega_max[0] = 0.0; 
  omega_max[1] = 0.0;
  omega_max[2] = cfac_ * sqrt(kymax_) * kzmax_;
}

//=======================================
// Linear_KS
// object for handling linear terms in KS
//=======================================
Linear_KS::Linear_KS(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  dB = dim3(min(128, grids_->Naky), 1, 1);
  dG = dim3(1+(grids_->Naky-1)/dB.x, 1, 1);
}

Linear_KS::~Linear_KS()
{
  // nothing
}

void Linear_KS::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) {

  // to be safe, start with zeros on RHS
  GRhs->set_zero();

  rhs_ks <<< dG, dB >>> (G->G(), GRhs->G(), grids_->ky, pars_->eps_ks);
}

//=======================================
// Linear_VP
// object for handling linear terms in VP
//=======================================
Linear_VP::Linear_VP(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  
  int nnx = grids_->Nyc;    int nbx = min(32, nnx);    int ngx = 1 + (nnx-1)/nbx;
  int nny = grids_->Nm;     int nby = min(16, nny);    int ngy = 1 + (nny-1)/nby;
  
  dB = dim3(nbx, nby, 1);
  dG = dim3(ngx, ngy, 1);
}

Linear_VP::~Linear_VP()
{
  // nothing
}

void Linear_VP::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) {

  // to be safe, start with zeros on RHS
  GRhs->set_zero();

  rhs_lin_vp <<< dG, dB >>> (G->G(), f->phi, GRhs->G(), grids_->ky,
			     pars_->vp_closure, pars_->vp_nu,       pars_->vp_nuh,
			     pars_->vp_alpha,   pars_->vp_alpha_h);


}
