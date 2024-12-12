#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
Lie_Trotter::Lie_Trotter(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv, const float* kperp2) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  geo_(geo), forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), bmagInv_(bmagInv), kperp2_(kperp2)
{
    // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  G_sm_s_phi = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  G_sm_s_apar = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  G_sm_b_apar = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

  G0 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  phi_l = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  apar_l = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

  mirror = (Cublas_test**) malloc(sizeof(void*)*grids_->Nspecies);
//  mirror = (Cusolve**) malloc(sizeof(void*)*grids_->Nspecies);

  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    A1[is] = new MomentsG (pars_, grids_, is_glob);
    A2[is] = new MomentsG (pars_, grids_, is_glob);
    A3[is] = new MomentsG (pars_, grids_, is_glob);
    B1[is] = new MomentsG (pars_, grids_, is_glob);
    B2[is] = new MomentsG (pars_, grids_, is_glob);
    B3[is] = new MomentsG (pars_, grids_, is_glob);
    G1[is] = new MomentsG (pars_, grids_, is_glob);
    
//    mirror[is] = new Cublas_test(pars_, grids_, geo_, 0.0, 1.0, 0.0, true, (double) pars_->dt, A1[is]->species->vt);
//    mirror[is] = new Cublas_test(pars_, grids_, geo_, 0.0, 1. + sqrtf(2.)/2., 0.0, true, (double) pars_->dt, A1[is]->species->vt);
    mirror[is] = new Cublas_test(pars_, grids_, geo_, pars_->p_, pars_->r_, pars_->u_, true, (double) pars_->dt, A1[is]->species->vt);

    checkCudaErrors(cudaGetLastError());




    if(pars_->implicit_preconditioner == "long_wavelength"){
      if(pars_->boundary_option_periodic){
        checkCuda(cudaMalloc((void**) &G_sm_s_phi[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
      }
      else{
        checkCuda(cudaMalloc((void**) &G_sm_s_phi[is],sizeof(cuComplex)*grids_->Nx*grids_->Nz*grids_->Nl*grids_->Nm));
      }
      if(pars_->fapar > 0){
	if(pars_->boundary_option_periodic){
          checkCuda(cudaMalloc((void**) &G_sm_s_apar[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
	}
	else{
	  checkCuda(cudaMalloc((void**) &G_sm_s_apar[is],sizeof(cuComplex)*grids_->Nx*grids_->Nz*grids_->Nl*grids_->Nm));
          checkCuda(cudaMalloc((void**) &G_sm_b_apar[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));

	}
      }
    }
    else{
      checkCuda(cudaMalloc((void**) &G_sm_s_phi[is],sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
      if(pars_->fapar > 0){
        checkCuda(cudaMalloc((void**) &G_sm_s_apar[is],sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));
      }
    }
    G0[is] = new MomentsG (pars_, grids_, is_glob);
    G2[is] = new MomentsG (pars_, grids_, is_glob);
    G3[is] = new MomentsG (pars_, grids_, is_glob);
    G4[is] = new MomentsG (pars_, grids_, is_glob);

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

  a21 = pars_->a21; a31 = pars_->a31; a32 = pars_->a32; w1 = pars_->w1, w2 = pars_->w2; w3 = pars_->w3;
  p_ = pars_->p_; q_ = pars_->q_; r_ = pars_->r_; s_ = pars_->s_; t_ = pars_->t_; u_ = pars_->u_;
  sdirk = pars_->sdirk;


  grad_par->zft_sherman_morrison_subsolve_lw(G_sm_s_phi, *(G1[0]->species), *(G1[ielectron]->species),r_*dt_,gradpar_, 0, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

  if(pars_->fapar > 0.){
    grad_par->zft_sherman_morrison_subsolve_lw(G_sm_s_apar, *(G1[0]->species), *(G1[ielectron]->species),r_*dt_,gradpar_, 1, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);
    set_mirror_apar_rhs<<<dG_m1, dB_m1>>>(G_sm_b_apar[ielectron], *(G1[ielectron]->species),geo_->bgrad, pars_->beta, r_*dt_); 
    mirror[ielectron]->invert_sherman_morrison(G_sm_b_apar[ielectron]);

  }


  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dB_lw = dim3(nt4, nt5, 1);
  dG_lw = dim3(nb4, nb5, 1); 

  dB_m1 = dim3(nt4, nt5, nt6);
  dG_m1 = dim3(nb4, nb5, nb6);  
  dB_m2 = dim3(nt1, nt2, nt4);
  dG_m2 = dim3(nb1, nb2, nb4);  

  flip = false;
}

Lie_Trotter::~Lie_Trotter()
{
  if (A1)    delete A1; 
  if (A2)    delete A2; 
  if (A3)    delete A3; 
  if (B1)    delete B1; 
  if (B2)    delete B2; 
  if (B3)    delete B3; 
  if (G1)    delete G1;
  if (G0)    delete G1; 

  if (grad_par) delete grad_par;
}

double Lie_Trotter::get_dt()
{
  if(pars_->flip_flop){
    return dt_;
  }
  else{
    return dt_;
  }
}

void Lie_Trotter::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron){
      linear_->rhs_nonstreaming_nonbounce(G[is], f, A[is], dt_);
//      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);

    }
    else{
      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
    }
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void Lie_Trotter::implicit_bounce(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
//      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);
      linear_->rhs_bounce(G[is], f, B[is], dt_);

    }
  }
}

void Lie_Trotter::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
//      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);
      linear_->rhs_streaming(G[is], f, B[is], dt_);

    }
    else{
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}


void Lie_Trotter::implicit_streaming(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron){
//      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);
      linear_->rhs_streaming(G[is], f, B[is], dt_);

    }
    else{
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}

void Lie_Trotter::invert_bounce(MomentsG** G1, Fields *f, double sdt)
{
  if(pars_->fapar > 0.){
    for(int is = 0; is < grids_->Nspecies; is++){
      G2[is]->copyFrom(G1[is]); 
    }
    G2[ielectron]->set_zero();
    solver_->fieldSolve(G2, f);
      
    add_apar_rhs<<<dG_m2, dB_m2>>>(G1[ielectron]->G(),f->apar,*(G1[ielectron]->species), sdt, geo_->bgrad);
  }
  for(int is = ielectron; is < grids_->Nspecies; is++){
    mirror[is]->invert_stream(G1[is]->G(), 0);

    if(pars_->fapar > 0.){
      sherman_morrison_mirror<<<dG_m2, dB_m2>>>(G1[is]->G(), G_sm_b_apar[is], solver_->getAmpereParFac()); 
    }
  } 
}

void Lie_Trotter::invert_streaming(MomentsG** G1, MomentsG** G0, Fields *f, double sdt,const float gradpar_, bool flip)
{

  int max_iter_streaming = pars_->implicit_max_iter_streaming;
  float omega_streaming = pars_->implicit_omega_streaming;
  for(int count = 0; count < max_iter_streaming; count++){
    if(count == 0){
      if(pars_->fapar > 0.){
        grad_par->zft_streaming_invert_full_em(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, false);

      }
      else{
        grad_par->zft_streaming_invert_full(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, false, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

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
	G1[is]->copyFrom(G0[is]);
      }
      
      if(pars_->fapar > 0.){
        grad_par->zft_streaming_invert_full_em(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, true);


      }
      else{
        grad_par->zft_streaming_invert_full(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, true, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);
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
}

void Lie_Trotter::ssprk3(MomentsG** A1, MomentsG** A2, MomentsG** A3, MomentsG** G, MomentsG** G1, Fields* f, bool setdt, float dt){
  explicit_terms(A1, G, f, false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], dt, A1[is]);
  }
  solver_->fieldSolve(G1,f);

  explicit_terms(A2,G1,f,false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], 1./4.*dt, A1[is], 1./4.*dt, A2[is]);
  }
  solver_->fieldSolve(G1,f);

  explicit_terms(A3,G1,f,false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], 1./6.*dt, A1[is], 1./6.*dt, A2[is], 2./3.*dt, A3[is]);
  }
  solver_->fieldSolve(G,f);


}

void Lie_Trotter::invert_implicit_terms_linked_lw(MomentsG** G1, MomentsG** G0, Fields *f, double sdt,const float gradpar_, int ielectron)
{
  int max_iter_streaming = pars_->implicit_max_iter_streaming;
  int max_iter_outer = pars_->implicit_max_iter;
  float omega_outer = pars_->implicit_omega;
  float omega_streaming = pars_->implicit_omega_streaming;

  grad_par->zft_sherman_morrison_subsolve_lw(G_sm_s_phi, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 0, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

  if(pars_->fapar > 0.){
    grad_par->zft_sherman_morrison_subsolve_lw(G_sm_s_apar, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 1, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_); 
    
    set_mirror_apar_rhs<<<dG_m1, dB_m1>>>(G_sm_b_apar[ielectron], *(G1[ielectron]->species),geo_->bgrad, pars_->beta, sdt); 
    mirror[ielectron]->invert_sherman_morrison(G_sm_b_apar[ielectron]);

  }

  for(int count_outer = 0; count_outer < max_iter_outer; count_outer++){
    if(count_outer != 0){
      solver_->fieldSolve(G1, f);
      implicit_terms(G2, G1, f);
      for(int is = 0; is < grids_->Nspecies; is++){
	G3[is]->copyFrom(G1[is]);
	G2[is]->add_scaled(1., G1[is], -sdt, G2[is]);
        G1[is]->add_scaled(1., G2[is], -1., G0[is]);
	G4[is]->copyFrom(G1[is]);
      }

    
    }

  for(int count_streaming = 0; count_streaming < max_iter_streaming; count_streaming++){
    if(count_streaming == 0){
      if(pars_->fapar > 0.){
        grad_par->zft_streaming_invert_full_em(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, false);
      }
      else{
        grad_par->zft_streaming_invert_full(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, false, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);


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
        grad_par->zft_streaming_invert_full_em(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, apar_l, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, pars_->beta, gradpar_, true);
      }
      else{
        grad_par->zft_streaming_invert_full(G1, G_sm_s_phi, G_sm_s_apar, G2, phi_l, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, true, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_);

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
      

    add_apar_rhs<<<dG_m2, dB_m2>>>(G1[ielectron]->G(),f->apar,*(G1[ielectron]->species), sdt, geo_->bgrad);
  }


  mirror[ielectron]->invert_stream(G1[ielectron]->G(), 0);
  if(pars_->fapar > 0.){
    sherman_morrison_mirror<<<dG_m2, dB_m2>>>(G1[ielectron]->G(), G_sm_b_apar[ielectron], solver_->getAmpereParFac()); 
  }

  if(count_outer != 0){
    for(int is = 0; is < grids_->Nspecies; is++){
      G1[is]->add_scaled(1.,G3[is], -1., G1[is]);
      G1[is]->add_scaled(omega_outer, G1[is], (1.-omega_outer), G3[is]);

    }
  }
  
  }

}


void Lie_Trotter::advance(double *t, MomentsG** G, Fields* f)
{
  float a = 1. + sqrtf(2.)/2.;
  float fac = 1.;
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }

  if(!flip){
    checkCudaErrors(cudaGetLastError()); 

    ssprk3(A1, A2, A3, G, G1, f, false, dt_/fac); 
    solver_->fieldSolve(G, f);       
    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

//    invert_implicit_terms_linked_lw(G, G0, f, dt_,gradpar_, ielectron);

    implicit_bounce(A1, G, f);
    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], q_*dt_, A1[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_bounce(G, f, r_*dt_/fac);
    solver_->fieldSolve(G, f);
    implicit_bounce(A2, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], s_*dt_, A1[is], t_*dt_, A2[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_bounce(G, f, u_*dt_/fac);
    solver_->fieldSolve(G, f);
    implicit_bounce(A3, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]);
    }


    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

    solver_->fieldSolve(G, f);

    implicit_terms(A1, G, f);
    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], q_*dt_, A1[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_streaming(G, G1, f, r_*dt_/fac,gradpar_, flip);
    solver_->fieldSolve(G, f);
    implicit_terms(A2, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], s_*dt_, A1[is], t_*dt_, A2[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_streaming(G, G1, f, u_*dt_/fac,gradpar_, flip);
    solver_->fieldSolve(G, f);
    implicit_terms(A3, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]);
    }

    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

    solver_->fieldSolve(G, f);
  }//FLIPPPPPP
  else{
    for(int is=0; is<grids_->Nspecies; is++) {
        G0[is]->copyFrom(G[is]);
    }

    implicit_terms(A1, G, f);
    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], q_*dt_, A1[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_streaming(G, G1, f, r_*dt_/fac,gradpar_, flip);
    solver_->fieldSolve(G, f);
    implicit_terms(A2, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], s_*dt_, A1[is], t_*dt_, A2[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_streaming(G, G1, f, u_*dt_/fac,gradpar_, flip);
    solver_->fieldSolve(G, f);
    implicit_terms(A3, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]);
    }
    
    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

    solver_->fieldSolve(G, f);

    implicit_bounce(A1, G, f);
    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], q_*dt_, A1[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_bounce(G, f, r_*dt_/fac);
    solver_->fieldSolve(G, f);
    implicit_bounce(A2, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], s_*dt_, A1[is], t_*dt_, A2[is]);
      G1[is]->copyFrom(G[is]);
    }

    invert_bounce(G, f, u_*dt_/fac);
    solver_->fieldSolve(G, f);
    implicit_bounce(A3, G, f);

    for(int is = 0; is < grids_->Nspecies; is++){
      G[is]->add_scaled(1., G0[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]);
    }


    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

    solver_->fieldSolve(G, f);       
    checkCudaErrors(cudaGetLastError()); 
    
    ssprk3(A1, A2, A3, G, G1, f, false, dt_/fac);
    for(int is=0; is<grids_->Nspecies; is++) {
      G0[is]->copyFrom(G[is]);
    }

    solver_->fieldSolve(G,f);

  }

/*  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  for (int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);*/
  checkCudaErrors(cudaGetLastError());
  *t += dt_;

  if(pars_->flip_flop){
    flip = !flip;
  }
}

