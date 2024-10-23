#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
IMEX_3stage_Full::IMEX_3stage_Full(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv, const float* kperp2) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), geo_(geo), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), bmagInv_(bmagInv), kperp2_(kperp2)
{
  
  // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  Gc = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  Gr = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  Ga = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

  G0 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  mirror = (Cublas_test**) malloc(sizeof(void*)*grids_->Nspecies);

  if(pars_->implicit_max_iter > 1){
    G3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
    G4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  }
  if(pars_->nstages > 3){
    A4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
    B4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  }


  phi_l = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  apar_l = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    A1[is] = new MomentsG (pars_, grids_, is_glob);
    A2[is] = new MomentsG (pars_, grids_, is_glob);
    A3[is] = new MomentsG (pars_, grids_, is_glob);
    B1[is] = new MomentsG (pars_, grids_, is_glob);
    B2[is] = new MomentsG (pars_, grids_, is_glob);
    B3[is] = new MomentsG (pars_, grids_, is_glob);
    G1[is] = new MomentsG (pars_, grids_, is_glob);

    mirror[is] = new Cublas_test(pars, grids, geo, pars->p_, pars->r_, pars->u_, pars->sdirk, (double) pars->dt, A1[is]->species->vt);


    if(pars_->nstages > 3){
      A4[is] = new MomentsG (pars_, grids_, is_glob);
      B4[is] = new MomentsG (pars_, grids_, is_glob);
    }

    if(pars_->implicit_preconditioner == "long_wavelength"){
      if(pars_->boundary_option_periodic){
        checkCuda(cudaMalloc((void**) &Gc[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
      }
      else{
        checkCuda(cudaMalloc((void**) &Gc[is],sizeof(cuComplex)*grids_->Nx*grids_->Nz*grids_->Nl*grids_->Nm));
      }
      if(pars_->fapar > 0){
	if(pars_->boundary_option_periodic){
          checkCuda(cudaMalloc((void**) &Gr[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
	}
	else{
	  checkCuda(cudaMalloc((void**) &Gr[is],sizeof(cuComplex)*grids_->Nx*grids_->Nz*grids_->Nl*grids_->Nm));
          checkCuda(cudaMalloc((void**) &Ga[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
	}
      }
    }
    else{
      checkCuda(cudaMalloc((void**) &Gc[is],sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
      if(pars_->fapar > 0){
        checkCuda(cudaMalloc((void**) &Gr[is],sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
      }
    }
    G0[is] = new MomentsG (pars_, grids_, is_glob);
    G2[is] = new MomentsG (pars_, grids_, is_glob);

    if(pars_->implicit_max_iter > 1){
      G3[is] = new MomentsG (pars_, grids_, is_glob);
      G4[is] = new MomentsG (pars_, grids_, is_glob);
    }
    checkCuda(cudaMalloc((void**) &phi_l[is], sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
    checkCuda(cudaMalloc((void**) &apar_l[is], sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
 
    // get species index of electrons
    if(pars_->species_h[is].type == 1) {
      ielectron = is;
      vte = A1[ielectron]->species->vt;
    }
  }

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
//  else if (pars_->boundary_option_periodic) {
//   grad_par = new GradParallelPeriodic(grids_);
//  }
  else {
    printf("USING GRADPARALLELLINKED!!!\n");
    grad_par = new GradParallelLinked(pars_, grids_);
  }

  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;

  int nn4 = grids_->Nz;              int nt4 = min(nn4,  16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nl;   int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn6 = grids_->Nm;   int nt6 = min(nn6,  4);   int nb6 = 1 + (nn6-1)/nt6;

  int nn7 = grids_->Nx*grids_->Nyc;   int nt7 = min(nn7,  8);   int nb7 = 1 + (nn7-1)/nt7;
  int nn8 = grids_->Nm*grids_->Nl;    int nt8 = min(nn8,  4);   int nb8 = 1 + (nn8-1)/nt8;

  a21 = pars_->a21; a31 = pars_->a31; a32 = pars_->a32; w1 = pars_->w1, w2 = pars_->w2; w3 = pars_->w3;
  p_ = pars_->p_; q_ = pars_->q_; r_ = pars_->r_; s_ = pars_->s_; t_ = pars_->t_; u_ = pars_->u_;
  sdirk = pars_->sdirk;

  nstages = pars_->nstages;
  a41 = pars_->a41, a42 = pars_->a42, a43 = pars_->a43, a44 = pars_->a44;
  w_ = pars_->w_, x_ = pars_->x_, y_ = pars_->y_, z_ = pars_->z_;
  w4 = pars_->w4;


  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dB_lw = dim3(nt4, nt5, 1);
  dG_lw = dim3(nb4, nb5, 1);  
  
  dB_m1 = dim3(nt4, nt5, nt6);
  dG_m1 = dim3(nb4, nb5, nb6);  
  dB_m2 = dim3(nt1, nt2, nt4);
  dG_m2 = dim3(nb1, nb2, nb4); 

  dB_all = dim3(nt7, nt4, nt8);
  dG_all = dim3(nb7, nb4, nb8);

}

IMEX_3stage_Full::~IMEX_3stage_Full()
{
  if (A1)    delete A1; 
  if (A2)    delete A2; 
  if (A3)    delete A3; 
  if (B1)    delete B1; 
  if (B2)    delete B2; 
  if (B3)    delete B3; 
  if (G1)    delete G1; 
  if (grad_par) delete grad_par;
}

void IMEX_3stage_Full::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron){
      linear_->rhs_nonstreaming_nonbounce(G[is], f, A[is], dt_);
    }
    else{
      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
    }
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void IMEX_3stage_Full::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);
    }
    else{
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}

void IMEX_3stage_Full::implicit_terms_id(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
      linear_->rhs_streaming_bounce_id(G[is], f, B[is], dt_,true);
    }
    else{
      linear_->rhs_streaming_id(G[is], f, B[is], dt_,true);
    }
  }
}

void IMEX_3stage_Full::apply_preconditioner(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
      linear_->rhs_bounce_id(G[is], f, B[is], dt_);

    }
    linear_->rhs_streaming_id(G[is], f, B[is], dt_,false);
  }
}


void IMEX_3stage_Full::invert_implicit_terms_linked_lw(MomentsG** G1, cuComplex** Gc, cuComplex** Gr, MomentsG** G0, MomentsG** G2, MomentsG** G3, MomentsG** G4, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron, int max_iter)
{
  int max_iter_streaming = pars_->implicit_max_iter_streaming;
//  int max_iter = pars_->implicit_max_iter;
  float omega = pars_->implicit_omega;
  float omega_streaming = pars_->implicit_omega_streaming;

  grad_par->zft_sherman_morrison_subsolve_lw(Gc, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 0, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

  if(pars_->fapar > 0.){
    grad_par->zft_sherman_morrison_subsolve_lw(Gr, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 1, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_); 
    
    set_mirror_apar_rhs<<<dG_m1, dB_m1>>>(Ga[ielectron], *(G1[ielectron]->species),geo_->bgrad, pars_->beta, sdt); 
    mirror[ielectron]->invert_sherman_morrison(Ga[ielectron]);

  }

  for(int count_outer = 0; count_outer < max_iter; count_outer++){
    if(count_outer != 0){
      solver_->fieldSolve(G1, f);
      implicit_terms(G2, G1, f);
      for(int is = 0; is < grids_->Nspecies; is++){
	G3[is]->copyFrom(G1[is]);
        negate_add_id<<<dG_all, dB_all>>>(G2[is]->G(), G1[is]->G(), sdt);
        G1[is]->add_scaled(1., G2[is], -1., G0[is]);
	G4[is]->copyFrom(G1[is]);
      }

    
    }

  for(int count = 0; count < max_iter_streaming; count++){
    if(count == 0){
      if(pars_->fapar > 0.){
        grad_par->zft_streaming_invert_full_em(G1, Gc, Gr, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, false);
      }
      else{
        grad_par->zft_streaming_invert_full(G1, Gc, Gr, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, false, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);


      }
      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
      }

    }
    else{
      solver_->fieldSolve(G1, f);
      for(int is = 0; is < grids_->Nspecies; is++){
	apply_flr_phi<<<dG, dB>>>(phi_l[is], f->phi, kperp2_, *(G1[is]->species));
	
	if(pars_->fapar > 0.){
	  apply_flr_phi<<<dG, dB>>>(apar_l[is], f->apar, kperp2_, *(G1[is]->species));
	}
        
	G2[is]->copyFrom(G1[is]);
	if(count_outer == 0){
	  G1[is]->copyFrom(G0[is]);
	}
	else{
	  G1[is]->copyFrom(G4[is]);
	}
      }
      
      if(pars_->fapar > 0.){
        grad_par->zft_streaming_invert_full_em(G1, Gc, Gr, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, true);
      }
      else{
//        grad_par->zft_streaming_invert_full(G1, Gc, Gr, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, true);
        grad_par->zft_streaming_invert_full(G1, Gc, Gr, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, true, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

      }
     
      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
	if (omega_streaming != 1.0){
          G1[is]->add_scaled(omega_streaming,G1[is], 1-omega_streaming,G2[is]);
	}
      }

      solver_->fieldSolve(G1, f);
      
    }
  }

  if(pars_->fapar > 0.){
    for(int is = 0; is < grids_->Nspecies; is++){
      G2[is]->copyFrom(G1[is]); 
    }
    G2[ielectron]->set_zero();
    solver_->fieldSolve(G2, f);
      
//    set_mirror_apar_rhs<<<dG_m1, dB_m1>>>(Gc[ielectron], *(G1[ielectron]->species),geo_->bgrad, pars_->beta, sdt); 
//    mirror[ielectron]->invert_sherman_morrison(Gc[ielectron]);

    add_apar_rhs<<<dG_m2, dB_m2>>>(G1[ielectron]->G(),f->apar,*(G1[ielectron]->species), sdt, geo_->bgrad);
  }


  mirror[ielectron]->invert_stream(G1[ielectron]->G(), 0);
  if(pars_->fapar > 0.){
    sherman_morrison_mirror<<<dG_m2, dB_m2>>>(G1[ielectron]->G(), Ga[ielectron], solver_->getAmpereParFac()); 
  }

  if(count_outer != 0){
    for(int is = 0; is < grids_->Nspecies; is++){
      G1[is]->add_scaled(1.,G3[is], -1., G1[is]);
      G1[is]->add_scaled(omega, G1[is], (1-omega), G3[is]);

    }
  }
  
  }

}



void IMEX_3stage_Full::invert_implicit_terms_linked(MomentsG** G1, cuComplex** Gc, cuComplex** Gr, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron)
{
  int max_iter = pars_->implicit_max_iter;
  float omega = pars_->implicit_omega;
  grad_par->zft_sherman_morrison_subsolve(Gc,solver_->get_max_Jflr(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_);

  for(int count = 0; count < max_iter; count++){
    if(count == 0){
      grad_par->zft_streaming_invert_full_laguerre(G1, Gc, Gr, G2, phi_l, solver_->get_max_qneutFacPhi_inv_l(), solver_->get_max_Jflr(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, false);
      
      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
      }

    }
    else{
      solver_->fieldSolve(G1, f);
      for(int is = 0; is < grids_->Nspecies; is++){
	apply_flr_phi<<<dG, dB>>>(phi_l[is], f->phi, kperp2_, *(G1[is]->species));
        
	G2[is]->copyFrom(G1[is]);
	G1[is]->copyFrom(G0[is]);
      }
      
      grad_par->zft_streaming_invert_full_laguerre(G1, Gc, Gr, G2, phi_l, solver_->get_max_qneutFacPhi_inv_l(), solver_->get_max_Jflr(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, true);
 
     
      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
	if (omega != 1.0){
          G1[is]->add_scaled(omega,G1[is], 1-omega,G2[is]);
	}
      }

      solver_->fieldSolve(G1, f);
      

    }
  }

  mirror[ielectron]->invert_stream(G1[ielectron]->G(), 0);

}


void IMEX_3stage_Full::invert_implicit_terms(MomentsG** G1, cuComplex** Gc, cuComplex** Gr, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron)
{
  int max_iter = pars_->implicit_max_iter;
  float omega = pars_->implicit_omega;
  for(int is = 0; is < grids_->Nspecies; is++){
    grad_par->zft(G1[is]);
  }
  sherman_morrison_subsolve_lw<<<dG_lw,dB_lw>>>(Gc[0],Gc[1], grids_->kz, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_,0);

  if(pars_->fapar > 0.){
    sherman_morrison_subsolve_lw<<<dG_lw,dB_lw>>>(Gr[0],Gr[1], grids_->kz, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 1);
 
  }
 
  for(int count = 0; count < max_iter; count++){
    if(count == 0){

      if(pars_->fapar > 0.0){
        tridiag_streaming_periodic_full_em<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), G2[0]->G(), G2[ielectron]->G(), phi_l[0], phi_l[ielectron], apar_l[0], apar_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, 0, false);
        sherman_morrison_full_em<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0], Gc[1], Gr[0], Gr[1], solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_);
      }
      else{
        tridiag_streaming_periodic_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), G2[0]->G(), G2[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, 0, false);
        sherman_morrison_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0], Gc[1], Gr[0], Gr[1], solver_->get_max_qneutFacPhi_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_);  
      }

      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
      }
    }
    else{
      solver_->fieldSolve(G1, f);
      for(int is = 0; is < grids_->Nspecies; is++){
	apply_flr_phi<<<dG, dB>>>(phi_l[is], f->phi, kperp2_, *(G1[is]->species));
	grad_par->zft_nmoms(phi_l[is], phi_l[is], grids_->Nl);

	if(pars_->fapar > 0.0){
	  apply_flr_phi<<<dG, dB>>>(apar_l[is], f->apar, kperp2_, *(G1[is]->species));
	  grad_par->zft_nmoms(apar_l[is], apar_l[is], grids_->Nl);
	}
        
	G2[is]->copyFrom(G1[is]);
	G1[is]->copyFrom(G0[is]);
	grad_par->zft(G1[is]);
	grad_par->zft(G2[is]);
      }

      if(pars_->fapar > 0.0){
        tridiag_streaming_periodic_full_em<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), G2[0]->G(), G2[ielectron]->G(), phi_l[0], phi_l[ielectron], apar_l[0], apar_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, 0, true);
        sherman_morrison_full_em<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0], Gc[1], Gr[0], Gr[1], solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_);
      }
      else{
        tridiag_streaming_periodic_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), G2[0]->G(), G2[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, 0, true);
        sherman_morrison_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0], Gc[1], Gr[0], Gr[1], solver_->get_max_qneutFacPhi_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_);  
      }
      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
	if(omega != 1.0){
          G1[is]->add_scaled(omega,G1[is], 1-omega,G2[is]);
	}
      }

    }
  }

  mirror[ielectron]->invert_stream(G1[ielectron]->G(), 0);


}

void IMEX_3stage_Full::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }

  // 3-stage IMEX methods
  // Explicit:
  //     ( 0    0    0  )
  //     ( a21  0    0  )
  //     ( a31  a32  0  )
  //     ----------------
  //     ( w1   w2   w3 )
  // Implicit:
  //     ( p    0    0  )
  //     ( q    r    0  )
  //     ( s    t    u  )
  //     ----------------
  //     ( w1   w2   w3 )
/*  double a21, a31, a32, w1, w2, w3;
  double p_, q_, r_, s_, t_, u_;
//  std::string scheme = "pareschi_russo_ssp2_332";
  std::string scheme = "conde_ssprk3_sdirk";
//  std::string scheme = "conde_3s3p";


  // Pareschi-Russo SSP2(3,3,2)
  if(scheme == "pareschi_russo_ssp2_332") {
    a21 = 0.5;
    a31 = 0.5;
    a32 = 0.5;
    w1 = 1./3.;
    w2 = 1./3.;
    w3 = 1./3.;
    p_ = 0.25;
    q_ = 0.;
    r_ = 0.25;
    s_ = 1./3.;
    t_ = 1./3.;
    u_ = 1./3.;
  } else if(scheme == "pareschi_russo_ssp2_322") {
    a21 = 0.;
    a31 = 0.;
    a32 = 1.;
    w1 = 0.;
    w2 = 1./2.;
    w3 = 1./2.;
    p_ = 0.5;
    q_ = -0.5;
    r_ = 0.5;
    s_ = 0.;
    t_ = 1./2.;
    u_ = 1./2.;
  } else if (scheme == "conde_ssprk3_sdirk") {
    a21 = 1.;
    a31 = 0.25;
    a32 = 0.25;
    w1 = 1./6.;
    w2 = 1./6.;
    w3 = 2./3.;
    p_ = 0.;
    q_ = 0.;
    r_ = 1.;
    s_ = 1./6.;
    t_ = -1./3.;
    u_ = 2./3.;
  } else if(scheme == "conde_3s3p") {
    a21 = 1.;
    a31 = 0.25;
    a32 = 0.25;
    w1 = 1./6.;
    w2 = 1./6.;
    w3 = 2./3.;
    p_ = 0.;
    q_ = (3. - sqrtf(3.))/6.;
    r_ = (3. + sqrtf(3.))/6.;
    s_ = (3. - sqrtf(3.))/24.;
    t_ = -(1. + sqrtf(3.))/8.;
    u_ = r_;
  } else if(scheme == "giraldo_ark2") {
    a21 = 2. - sqrtf(2.);
    a32 = (3. + 2.*sqrtf(2.))/6.;
    a31 = 1. - a32;
    w1 = 1./sqrtf(8.);
    w2 = 1./sqrtf(8.);
    w3 = 1.-1./sqrtf(2.);
    p_ = 0.;
    q_ = 1. - 1./sqrtf(2.);
    r_ = 1. - 1./sqrtf(2.);
    s_ = 1./sqrtf(8.);
    t_ = 1./sqrtf(8.);
    u_ = 1.-1./sqrtf(2.);
  }*/
  checkCudaErrors(cudaGetLastError()); 
  // stage 1
  for (int is=0; is<grids_->Nspecies; is++) {
    G1[is]->copyFrom(G[is]);
  }
  if(p_!=0.) {
    // compute Phi1_i (with G1_e=0)
    solver_->fieldSolve(G1, f);
    G1[ielectron]->copyFrom(G[ielectron]);
    // G1_e = inv(I - p_*dt*B)*G1_e
    if(pars_->boundary_option_periodic){
      invert_implicit_terms(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, p_*dt_,gradpar_, bmagInv_, ielectron);
    }
    else{
      if(pars_->implicit_preconditioner == "long_wavelength"){
        invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, G3, G4, f, phi_l, apar_l, p_*dt_,gradpar_, bmagInv_, ielectron, 1);
//        invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, p_*dt_,gradpar_, bmagInv_, ielectron, false);

      }
      else{
        invert_implicit_terms_linked(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, p_*dt_,gradpar_, bmagInv_, ielectron);
      }
    }

    solver_->fieldSolve(G1, f);
    if (pars_->dealias_kz) grad_par->dealias(f->phi);
  }

  // stage 2
  // compute A1 = A(G1)
  explicit_terms(A1, G1, f, false);
  // compute B1 = B(G1)
  implicit_terms(B1, G1, f);
  // G1_i = G_i + a21*dt*A1_i

  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], a21*dt_, A1[is], q_*dt_, B1[is]);
    G0[is]->copyFrom(G1[is]);
  }
  // G1_e = inv(I - r_*dt*B)*G1_e
  if(pars_->boundary_option_periodic){
//  if(true){
    invert_implicit_terms(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, r_*dt_,gradpar_, bmagInv_, ielectron);
  }
  else{
    if(pars_->implicit_preconditioner == "long_wavelength"){
      invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, G3, G4, f, phi_l, apar_l, r_*dt_,gradpar_, bmagInv_, ielectron, pars_->implicit_max_iter);
//        invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, r_*dt_,gradpar_, bmagInv_, ielectron, false);

    }
    else{
      invert_implicit_terms_linked(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, r_*dt_,gradpar_, bmagInv_, ielectron);
    }
  }
  solver_->fieldSolve(G1, f);        
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  // compute A2 = A(G1)
  explicit_terms(A2, G1, f, false);
  // compute B2 = B(G1)
  implicit_terms(B2, G1, f);
  // G1_i = G_i + a31*A1_i + a32*A2_i + s_*dt*B1_i + t_*dt*B2_i
  for (int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], a31*dt_, A1[is], a32*dt_, A2[is], s_*dt_, B1[is], t_*dt_, B2[is]);
    G0[is]->copyFrom(G1[is]);
  }
  // G1 = inv(I - u_*dt*B)*G1
  if(pars_->boundary_option_periodic){
//  if(true){
    invert_implicit_terms(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron);
  }
  else{
    if(pars_->implicit_preconditioner == "long_wavelength"){
      invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, G3, G4, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron, pars_->implicit_max_iter);
//        invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_, bmagInv_, ielectron, false);

    }
    else{
      invert_implicit_terms_linked(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron);
    }
  }
  
  solver_->fieldSolve(G1, f);          
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  // combine stage
  // compute A3 = A(G1)
  explicit_terms(A3, G1, f, false);
  // compute B3 = B(G1)
  implicit_terms(B3, G1, f);


  if(nstages > 3){
  // stage 4
  // G1_i = G_i + a31*A1_i + a32*A2_i + s_*dt*B1_i + t_*dt*B2_i
  for (int is=0; is<grids_->Nspecies; is++) {
      G1[is]->add_scaled(1., G[is], a41*dt_, A1[is], a42*dt_, A2[is], a43*dt_, A3[is]);
      G1[is]->add_scaled(1., G1[is], w_*dt_, B1[is], x_*dt_, B2[is], y_*dt_, B3[is]);
      G0[is]->copyFrom(G1[is]);
  }
 
  // G1 = inv(I - u_*dt*B)*G1

  if(pars_->boundary_option_periodic){
//  if(true){
    invert_implicit_terms(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron);
  }
  else{
    if(pars_->implicit_preconditioner == "long_wavelength"){
      invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, G3, G4, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron, pars_->implicit_max_iter);
//        invert_implicit_terms_linked_lw(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_, bmagInv_, ielectron, false);

    }
    else{
      invert_implicit_terms_linked(G1, Gc, Gr, G0, G2, f, phi_l, apar_l, u_*dt_,gradpar_,bmagInv_, ielectron);
    }
  }

  solver_->fieldSolve(G1, f);          
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  // combine stage
  // compute A3 = A(G1)
  explicit_terms(A4, G1, f, false);
  // compute B3 = B(G1)
  implicit_terms(B4, G1, f);

  // G = G + w1*A1 + w2*A2 + w3*A3 + w1*B1 + w2*B2 + w3*B3
  for (int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is], w4*dt_, A4[is]); 
    G[is]->add_scaled(1., G[is], w1*dt_, B1[is], w2*dt_, B2[is], w3*dt_, B3[is], w4*dt_, B4[is]); 
  }
  }
  else{
   for (int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]);
    G[is]->add_scaled(1., G[is], w1*dt_, B1[is], w2*dt_, B2[is], w3*dt_, B3[is]); 
  }
   
  }

  solver_->fieldSolve(G, f);        
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  for (int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  *t += dt_;
  checkCudaErrors(cudaGetLastError());
}
