#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
IMEX_3stage::IMEX_3stage(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), geo_(geo), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), bmagInv_(bmagInv)
{
  
  // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 

  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  B4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 

  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gc = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gr = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  Gr2 = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);
  Ga = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

  mirror = (Cublas_test**) malloc(sizeof(void*)*grids_->Nspecies);


  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    A1[is] = new MomentsG (pars_, grids_, is_glob);
    A2[is] = new MomentsG (pars_, grids_, is_glob);
    A3[is] = new MomentsG (pars_, grids_, is_glob);
    A4[is] = new MomentsG (pars_, grids_, is_glob);

    B1[is] = new MomentsG (pars_, grids_, is_glob);
    B2[is] = new MomentsG (pars_, grids_, is_glob);
    B3[is] = new MomentsG (pars_, grids_, is_glob);
    B4[is] = new MomentsG (pars_, grids_, is_glob);

    G1[is] = new MomentsG (pars_, grids_, is_glob);
    Gc[is] = new MomentsG (pars_, grids_, is_glob);
    Gr[is] = new MomentsG (pars_, grids_, is_glob);
    G2[is] = new MomentsG (pars_, grids_, is_glob);
    G3[is] = new MomentsG (pars_, grids_, is_glob);
    G4[is] = new MomentsG (pars_, grids_, is_glob);

    checkCuda(cudaMalloc((void**) &Gr2[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));
    checkCuda(cudaMalloc((void**) &Ga[is],sizeof(cuComplex)*grids_->Nz*grids_->Nl*grids_->Nm));


    mirror[is] = new Cublas_test(pars, grids, geo, pars->p_, pars->r_, pars->u_, pars->sdirk, (double) pars->dt, A1[is]->species->vt);

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

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dB_m1 = dim3(nt4, nt5, nt6);
  dG_m1 = dim3(nb4, nb5, nb6);  
  dB_m2 = dim3(nt1, nt2, nt4);
  dG_m2 = dim3(nb1, nb2, nb4); 

  dB_all = dim3(nt7, nt4, nt8);
  dG_all = dim3(nb7, nb4, nb8);


  if(pars_->fapar > 0.){
//    grad_par->zft_sherman_morrison_subsolve_lw(Gr2, *(G1[0]->species), *(G1[ielectron]->species),sdt,gradpar_, 1, pars_->hypercollisions_const, pars_->hypercollisions_kz, pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->nu_hyper_lm, pars_->p_hyper_l, pars_->p_hyper_m, pars_->p_hyper_lm, pars_->vtmax, dt_); 
    
    set_mirror_apar_rhs<<<dG_m1, dB_m1>>>(Ga[ielectron], *(G1[ielectron]->species),geo_->bgrad, pars_->beta, pars_->r_*dt_); 
    mirror[ielectron]->invert_sherman_morrison(Ga[ielectron]);
  }

}

IMEX_3stage::~IMEX_3stage()
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

void IMEX_3stage::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron) {
      linear_->rhs_nonstreaming_nonbounce(G[is], f, A[is], dt_);
    } else {
      linear_->rhs(G[is], f, A[is], dt_);
      }
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void IMEX_3stage::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron) { // electrons
      // compute implicit part of electron linear rhs
      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);

    }
  }
}

void IMEX_3stage::invert_implicit_terms(MomentsG** G1, MomentsG* Gc, MomentsG** Gr, MomentsG** G2, MomentsG** G3, MomentsG** G4, cuComplex** Gr2, cuComplex** Ga, Fields *f, double sdt,const float gradpar_, const float* bmagInv_, int ielectron)
{


  double sdtvt = sdt*vte;
  // tridiag from numerical recipes
//  int max_iter = pars_->implicit_max_iter
  for(int count_outer = 0; count_outer < pars_->implicit_max_iter; count_outer++){
    if(count_outer != 0){
      solver_->fieldSolve(G1, f);
      implicit_terms(G2, G1, f);
      G3[ielectron]->copyFrom(G1[ielectron]);
      negate_add_id<<<dG_all, dB_all>>>(G2[ielectron]->G(), G1[ielectron]->G(), sdt);
      G1[ielectron]->add_scaled(1., G2[ielectron], -1., Gc);
      G4[ielectron]->copyFrom(G1[ielectron]);
    }

  if(pars_->local_limit) {
    // FFT Phi_i
    grad_par->zft(f->phi, f->phi);
    grad_par->zft(f->apar, f->apar);

    // FFT G_e
    grad_par->zft(G1[ielectron]);
    if (pars_-> fapar > 0.0){
     tridiag_streaming_local_em<<<dG, dB>>>(G1[ielectron]->G(), f->phi, f->apar, 1./grids_->Zp, solver_->getQneutDenom(), solver_->getAmpereParFac(), *(G1[ielectron]->species), sdtvt, pars_->beta);  
    }
    else{
    tridiag_streaming_local<<<dG, dB>>>(G1[ielectron]->G(), f->phi, 1./grids_->Zp, solver_->getQneutDenom(), *(G1[ielectron]->species), sdtvt);
    }
  } //else if (pars_->boundary_option_periodic) {
  else if(false){
      // FFT Phi_i
      grad_par->zft(f->phi, f->phi);
      grad_par->zft(f->apar, f->apar);

      // FFT G_e
      grad_par->zft(G1[ielectron]);

      int max_iter = pars_->implicit_max_iter_streaming;
      double omega = pars_->implicit_omega_streaming;
      for (int count = 0; count < max_iter; count++){
        if (count == 0){ //This is just using J(z=0)
	  if (pars_-> fapar > 0.0){
	    tridiag_streaming_periodic_em<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(), f->phi, f->apar, grids_->kz, solver_->getQneutDenom(), solver_->getAmpereParFac(), *(G1[ielectron]->species), sdtvt, gradpar_, false, pars_->beta);
	  }
	  else if (pars_->fbpar > 0.0){
            grad_par->zft(f->bpar, f->bpar);
	    tridiag_streaming_periodic_bpar<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(), f->phi, f->apar, f->bpar, grids_->kz, solver_->get_max_qneutFacPhi_inv(), solver_->get_max_ampereParFac_inv(), solver_->get_max_qneutFacBpar_inv(), solver_->get_max_amperePerpFacPhi_inv(), solver_->get_max_amperePerpFacBpar_inv(),solver_->getBparDenom(), *(G1[ielectron]->species), sdtvt, gradpar_, bmagInv_, false, pars_->beta);

	  }
	  else{
            tridiag_streaming_periodic<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(), f->phi, grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[ielectron]->species), sdtvt, gradpar_, false);
	  }
        }
        else{
          grad_par->zft_inverse(G1[ielectron]); //Calculate full potential
          Gr[ielectron]->copyFrom(G1[ielectron]);
          solver_->fieldSolve(G1, f);
          grad_par->zft(f->phi, f->phi);

          G1[ielectron]->copyFrom(Gc); //I think for iteration scheme, need original G1
          grad_par->zft(G1[ielectron]);
          grad_par->zft(Gr[ielectron]);
          tridiag_streaming_periodic<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(),f->phi, grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[ielectron]->species), sdtvt, gradpar_, true);

          grad_par->zft_inverse(G1[ielectron]);
          grad_par->zft_inverse(Gr[ielectron]);
          G1[ielectron]->add_scaled(omega,G1[ielectron],(1.-omega),Gr[ielectron]);
          grad_par->zft(G1[ielectron]);
        }
      }

  } else if (!pars_->boundary_option_periodic && !pars_->implicit_linked){
      // FFT Phi_i
      grad_par->zft(f->phi, f->phi);
      // FFT G_e
      grad_par->zft(G1[ielectron]);

      int max_iter = pars_->implicit_max_iter_streaming;
      double omega = pars_->implicit_omega_streaming;
      for (int count = 0; count < max_iter; count++){
        if (count == 0){ //This is just using J(z=0)
          tridiag_streaming_periodic<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(), f->phi, grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[ielectron]->species), sdtvt, gradpar_, false);
        }
        else{
      	  grad_par->zft_inverse(G1[ielectron]); //Calculate full potential
	  Gr[ielectron]->copyFrom(G1[ielectron]);
    	  solver_->fieldSolve(G1, f);
  	  grad_par->zft(f->phi, f->phi);

	  G1[ielectron]->copyFrom(Gc); //I think for iteration scheme, need original G1
	  grad_par->zft(G1[ielectron]);
	  grad_par->zft(Gr[ielectron]);
  	  tridiag_streaming_periodic<<<dG, dB>>>(G1[ielectron]->G(), Gr[ielectron]->G(),f->phi, grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[ielectron]->species), sdtvt, gradpar_, true);

	  grad_par->zft_inverse(G1[ielectron]); 
	  grad_par->zft_inverse(Gr[ielectron]);
	  G1[ielectron]->add_scaled(omega,G1[ielectron],(1.-omega),Gr[ielectron]); 
	  grad_par->zft(G1[ielectron]);
        }          
      }
  
    }
    else{
      int max_iter = pars_->implicit_max_iter_streaming;
      double omega = pars_->implicit_omega_streaming;
/*      for(int is = 0; is < grids_->Nspecies; is++){
        G2[is]->copyFrom(G1[is]); 
      }
      G2[ielectron]->set_zero();
      solver_->fieldSolve(G2, f);*/
/*      if(count_outer != 0){
        solver_->fieldSolve(G1, f);
      }*/

      if(count_outer != 0){
	for(int is = 0; is < grids_->Nspecies; is++){
          G2[is]->set_zero();
	}
        solver_->fieldSolve(G2, f);

      }


      for (int count = 0; count < max_iter; count++){
	if (count ==0){
          Gr[ielectron]->copyFrom(G1[ielectron]);
          checkCudaErrors(cudaGetLastError());
	  if(pars_->fapar > 0. || pars_->fbpar > 0.){
            checkCudaErrors(cudaGetLastError());
//	    grad_par->zft_streaming_invert_em(G1[ielectron], Gr[ielectron],f->phi,f->apar,f->bpar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_qneutFacBpar_inv(),solver_->get_max_ampereParFac_inv(),solver_->get_max_amperePerpFacPhi_inv(),solver_->get_max_amperePerpFacBpar_inv(),sdtvt, gradpar_, false);
	    grad_par->zft_streaming_invert_apar(G1[ielectron], Gr[ielectron],f->phi,f->apar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_ampereParFac_inv(), sdtvt, gradpar_, false, pars_->hypercollisions_kz, pars_->p_hyper_m, pars_->nu_hyper_m);


	  }
	  else{
            grad_par->zft_streaming_invert(G1[ielectron], Gr[ielectron], f->phi,solver_->getQneutDenom(),solver_->get_max_qneutFacPhi_inv(),sdtvt, gradpar_, false); 
	  }
	}
	else{
	  grad_par->zft_inverse(G1[ielectron]); //Calculate full potential
          Gr[ielectron]->copyFrom(G1[ielectron]);
          solver_->fieldSolve(G1, f);

//          G1[ielectron]->copyFrom(Gc); //I think for iteration scheme, need original G1
	  if(count_outer == 0){
	    G1[ielectron]->copyFrom(Gc);
	  }
	  else{
	    G1[ielectron]->copyFrom(G4[ielectron]);
	  }


          if(pars_->fapar > 0. || pars_->fbpar > 0.){
//            grad_par->zft_streaming_invert_em(G1[ielectron], Gr[ielectron],f->phi,f->apar,f->bpar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_qneutFacBpar_inv(),solver_->get_max_ampereParFac_inv(),solver_->get_max_amperePerpFacPhi_inv(),solver_->get_max_amperePerpFacBpar_inv(),sdtvt, gradpar_, true);
	    grad_par->zft_streaming_invert_apar(G1[ielectron], Gr[ielectron],f->phi,f->apar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_ampereParFac_inv(), sdtvt, gradpar_, true, pars_->hypercollisions_kz, pars_->p_hyper_m, pars_->nu_hyper_m);


          }
          else{
            grad_par->zft_streaming_invert(G1[ielectron], Gr[ielectron], f->phi,solver_->getQneutDenom(),solver_->get_max_qneutFacPhi_inv(),sdtvt, gradpar_, true);
          }

          G1[ielectron]->add_scaled(omega,G1[ielectron],(1.-omega),Gr[ielectron]);
	}
      }
    }

  grad_par->zft_inverse(G1[ielectron]);

  if(pars_->fapar > 0.){
    if(count_outer == 0){
    for(int is = 0; is < grids_->Nspecies; is++){
      G2[is]->copyFrom(G1[is]); 
    }
    G2[ielectron]->set_zero();
    solver_->fieldSolve(G2, f);
    }
    else{
      for(int is = 0; is < grids_->Nspecies; is++){
        G2[is]->set_zero();
      }
      solver_->fieldSolve(G2, f);
    }
    add_apar_rhs<<<dG_m2, dB_m2>>>(G1[ielectron]->G(),f->apar,*(G1[ielectron]->species), sdt, geo_->bgrad);
  }


  mirror[ielectron]->invert_stream(G1[ielectron]->G(), 0);
  if(pars_->fapar > 0.){
    sherman_morrison_mirror<<<dG_m2, dB_m2>>>(G1[ielectron]->G(), Ga[ielectron], solver_->getAmpereParFac()); 
  }
  if(count_outer != 0){
      G1[ielectron]->add_scaled(1.,G3[ielectron], -1., G1[ielectron]);
  }
  }

}

void IMEX_3stage::advance(double *t, MomentsG** G, Fields* f)
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
  double a21, a31, a32, a41, a42, a43, w1, w2, w3, w4;
  double p_, q_, r_, s_, t_, u_, w_, x_, y_, z_;
  nstages = 3;
//  std::string scheme = "giraldo_ark2";
//  std::string scheme = "conde_ssprk3_sdirk";
//  std::string scheme = "conde_3s3p";
//  std::string scheme = "pareschi_russo_ssp2_322";
  std::string scheme = "sundials_ark3";
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
  } else if (scheme == "sundials_ark3"){
      a21 = 1767732205903./2027836641118.;
      a31 = 5535828885825./10492691773637.;
      a32 = 788022342437./10882634858940.;
      a41 = 6485989280629./16251701735622.;
      a42 = -4246266847089./9704473918619.;
      a43 = 10755448449292./10357097424841.;

      p_ = 0.;
      q_ = 1767732205903./4055673282236.;
      r_ = 1767732205903./4055673282236.;
      s_ = 2746238789719./10658868560708.;
      t_ = -640167445237./6845629431997.;
      u_ = 1767732205903./4055673282236.;
      w_ = 1471266399579./7840856788654.;
      x_ = -4482444167858./7529755066697.;
      y_ = 11266239266428./11593286722821.;
      z_ = 1767732205903./4055673282236.;
      
      w1 = 1471266399579./7840856788654.;
      w2 = -4482444167858./7529755066697.;
      w3 = 11266239266428./11593286722821.;
      w4 = 1767732205903./4055673282236.;
      nstages = 4;
  }
  checkCudaErrors(cudaGetLastError()); 
  // stage 1
  for (int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron && p_!=0.) {
      G1[is]->set_zero();
    } else {
      G1[is]->copyFrom(G[is]);
    }
  }
  if(p_!=0.) {
    // compute Phi1_i (with G1_e=0)
    solver_->fieldSolve(G1, f);
    G1[ielectron]->copyFrom(G[ielectron]);
    // G1_e = inv(I - p_*dt*B)*G1_e
//    invert_implicit_terms(G1[ielectron], f, p_*dt_,gradpar_);
    Gc[ielectron]->copyFrom(G1[ielectron]);
    Gr[ielectron]->copyFrom(G1[ielectron]);
//    invert_implicit_terms(G1, Gc[ielectron], Gr, f, p_*dt_,gradpar_, bmagInv_, ielectron);
    invert_implicit_terms(G1, Gc[ielectron], Gr, G2, G3, G4, Gr2, Ga, f, p_*dt_,gradpar_, bmagInv_, ielectron);


    solver_->fieldSolve(G1, f);
    if (pars_->dealias_kz) grad_par->dealias(f->phi);
  }
  // stage 2
  // compute A1 = A(G1)
  //the following is a shitty way to compute the ion contribution to phi.  
  explicit_terms(A1, G1, f, false);
  // compute B1 = B(G1)
  implicit_terms(B1, G1, f);
  // G1_i = G_i + a21*dt*A1_i

  for(int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron) { // electrons
      G1[is]->set_zero();
    } else {
      G1[is]->add_scaled(1., G[is], a21*dt_, A1[is]);
    }
  }
  // compute Phi1_i (with G1_e=0)
  solver_->fieldSolve(G1, f);         
  // G1_e = G_e + a21*dt*A1_e + q_*dt*B1_e
  G1[ielectron]->add_scaled(1., G[ielectron], a21*dt_, A1[ielectron], q_*dt_, B1[ielectron]);
  // G1_e = inv(I - r_*dt*B)*G1_e
//  invert_implicit_terms(G1[ielectron], f, r_*dt_,gradpar_);
  Gc[ielectron]->copyFrom(G1[ielectron]);
  Gr[ielectron]->copyFrom(G1[ielectron]);
//  invert_implicit_terms(G1, Gc[ielectron], Gr, f, r_*dt_,gradpar_, bmagInv_, ielectron);
  invert_implicit_terms(G1, Gc[ielectron], Gr, G2, G3, G4, Gr2, Ga, f, r_*dt_,gradpar_, bmagInv_, ielectron);


  solver_->fieldSolve(G1, f);        
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  // compute A2 = A(G1)
  explicit_terms(A2, G1, f, false);
  // compute B2 = B(G1)

  implicit_terms(B2, G1, f);

  // G1_i = G_i + a31*A1_i + a32*A2_i + s_*dt*B1_i + t_*dt*B2_i
  for (int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron) { // electrons
      G1[is]->set_zero();
    } else {
      G1[is]->add_scaled(1., G[is], a31*dt_, A1[is], a32*dt_, A2[is], s_*dt_, B1[is], t_*dt_, B2[is]);
    }
  }
 
  // compute Phi_i (with G1_e=0)
  solver_->fieldSolve(G1, f);        
  // G1_e = G_e + a31*A1_e + a32*A2_e + s_*dt*B1_e + t_*dt*B2_e
  G1[ielectron]->add_scaled(1., G[ielectron], a31*dt_, A1[ielectron], a32*dt_, A2[ielectron], 
		            s_*dt_, B1[ielectron], t_*dt_, B2[ielectron]);
  // G1 = inv(I - u_*dt*B)*G1
//  invert_implicit_terms(G1[ielectron], f, u_*dt_,gradpar_);
  Gc[ielectron]->copyFrom(G1[ielectron]);
  Gr[ielectron]->copyFrom(G1[ielectron]);

//  invert_implicit_terms(G1, Gc[ielectron], Gr, f, u_*dt_,gradpar_,bmagInv_, ielectron);
  invert_implicit_terms(G1, Gc[ielectron], Gr, G2, G3, G4, Gr2, Ga, f, u_*dt_,gradpar_, bmagInv_, ielectron);

  
  solver_->fieldSolve(G1, f);          
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  // combine stage
  // compute A3 = A(G1)
  explicit_terms(A3, G1, f, false);
  // compute B3 = B(G1)
  implicit_terms(B3, G1, f);
  // G = G + w1*A1 + w2*A2 + w3*A3 + w1*B1 + w2*B2 + w3*B3

  if(nstages > 3){
  // stage 4
  // G1_i = G_i + a31*A1_i + a32*A2_i + s_*dt*B1_i + t_*dt*B2_i
  for (int is=0; is<grids_->Nspecies; is++) {
      if(is == ielectron){
        G1[is]->set_zero();
      }
      else{
        G1[is]->add_scaled(1., G[is], a41*dt_, A1[is], a42*dt_, A2[is], a43*dt_, A3[is]);
        G1[is]->add_scaled(1., G1[is], w_*dt_, B1[is], x_*dt_, B2[is], y_*dt_, B3[is]);
      }
  }

  // compute Phi_i (with G1_e=0)
  solver_->fieldSolve(G1, f);        
  // G1_e = G_e + a31*A1_e + a32*A2_e + s_*dt*B1_e + t_*dt*B2_e
  
  G1[ielectron]->add_scaled(1., G[ielectron], a41*dt_, A1[ielectron], a42*dt_, A2[ielectron], a43*dt_, A3[ielectron]);
  G1[ielectron]->add_scaled(1., G1[ielectron], w_*dt_, B1[ielectron], x_*dt_, B2[ielectron], y_*dt_, B3[ielectron]);

  // G1 = inv(I - u_*dt*B)*G1
//  invert_implicit_terms(G1[ielectron], f, u_*dt_,gradpar_);
  Gc[ielectron]->copyFrom(G1[ielectron]);
  Gr[ielectron]->copyFrom(G1[ielectron]);

//  invert_implicit_terms(G1, Gc[ielectron], Gr, f, z_*dt_,gradpar_,bmagInv_, ielectron);
  invert_implicit_terms(G1, Gc[ielectron], Gr, G2, G3, G4, Gr2, Ga, f, z_*dt_,gradpar_, bmagInv_, ielectron);



  // G1 = inv(I - u_*dt*B)*G1
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
// ======= 4-stage addivte RK IMEX methods =======
IMEX_4stage::IMEX_4stage(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1)
{
  
  // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  B4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    A1[is] = new MomentsG (pars_, grids_, is);
    A2[is] = new MomentsG (pars_, grids_, is);
    A3[is] = new MomentsG (pars_, grids_, is);
    A4[is] = new MomentsG (pars_, grids_, is);
    B1[is] = new MomentsG (pars_, grids_, is);
    B2[is] = new MomentsG (pars_, grids_, is);
    B3[is] = new MomentsG (pars_, grids_, is);
    B4[is] = new MomentsG (pars_, grids_, is);
    G1[is] = new MomentsG (pars_, grids_, is);
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
//    grad_par = new GradParallelPeriodic(grids_);
//  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }
  
  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);
  nstage = 4;
}
IMEX_4stage::~IMEX_4stage()
{
  if (A1)    delete A1; 
  if (A2)    delete A2; 
  if (A3)    delete A3; 
  if (A4)    delete A4; 
  if (B1)    delete B1; 
  if (B2)    delete B2; 
  if (B3)    delete B3; 
  if (B4)    delete B4; 
  if (G1)    delete G1; 
  if (grad_par) delete grad_par;
}
void IMEX_4stage::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron) { // electrons
      // compute explicit part of electron linear rhs
      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
    } else {
      // handle entire ion linear rhs explicitly
      linear_->rhs(G[is], f, A[is], dt_);
    }
    // compute nonlinear terms explicitly for all species
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
    }
  }
}
void IMEX_4stage::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron) { // electrons
      // compute implicit part of electron linear rhs
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}
void IMEX_4stage::invert_implicit_terms(MomentsG* G1e, Fields* f, double sdt)
{
  // FFT Phi_i
  grad_par->zft(f->phi, f->phi);
  // FFT G_e
  grad_par->zft(G1e);
  double sdtvt = sdt*vte;
  // tridiag from numerical recipes
  if(pars_->local_limit) {
    tridiag_streaming_local<<<dG, dB>>>(G1e->G(), f->phi, 1./grids_->Zp, solver_->getQneutDenom(), *(G1e->species), sdtvt);
  } else if (pars_->boundary_option_periodic) {
//    tridiag_streaming_periodic<<<dG, dB>>>(G1e->G(), f->phi, grids_->kz, solver_->getQneutDenom(), *(G1e->species), sdtvt);
  } else {
//    tridiag_streaming_periodic<<<dG, dB>>>(G1e->G(), f->phi, grids_->kz, solver_->getQneutDenom(), *(G1e->species), sdtvt);
  }
  grad_par->zft_inverse(G1e);
}
void IMEX_4stage::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }
  // 4-stage IMEX methods
  // Explicit:
  //     ( 0    0    0    0  )
  //     ( a21  0    0    0  )
  //     ( a31  a32  0    0  )
  //     ( a41  a42  a43  0  )
  //     ---------------------
  //     ( w1   w2   w3   w4 )
  // Implicit:
  //     ( b11  0    0    0   )
  //     ( b21  b22  0    0   )
  //     ( b31  b32  b33  0   )
  //     ( b41  b42  b43  b44 )
  //     ----------------------
  //     ( w1   w2   w3   w4  )
  double a21;
  double a31, a32;
  double a41, a42, a43;
  double w1, w2, w3, w4;
  double b11;
  double b21, b22;
  double b31, b32, b33;
  double b41, b42, b43, b44;
  std::string scheme = "kennedy_carpenter_ark3";
  if(scheme == "kennedy_carpenter_ark3") {
    a21 = 1767732205903./2027836641118.;
    a31 = 5535828885825./10492691773637.;
    a32 = 788022342437./10882634858940.;
    a41 = 6485989280629./16251701735622.;
    a42 = -4246266847089./9704473918619.;
    a43 = 10755448449292./10357097424841.;
    w1 = 1471266399579./7840856788654.;
    w2 = -4482444167858./7529755066697.;
    w3 = 11266239266428./11593286722821.;
    w4 = 1767732205903./4055673282236.;
    b11 = 0.;
    b21 = 1767732205903./4055673282236.;
    b22 = 1767732205903./4055673282236.;
    b31 = 2746238789719./10658868560708.;
    b32 = -640167445237./6845629431997.;
    b33 = 1767732205903./4055673282236.;
    b41 = 1471266399579./7840856788654.;
    b42 = -4482444167858./7529755066697.;
    b43 = 11266239266428./11593286722821.;
    b44 = 1767732205903./4055673282236.;
  } 
  
  // stage 1
  for (int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron && b11!=0.) {
      G1[is]->set_zero();
    } else {
      G1[is]->copyFrom(G[is]);
    }
  }
  if(b11!=0.) {
    // compute Phi1_i (with G1_e=0)
    solver_->fieldSolve(G1, f);
    G1[ielectron]->copyFrom(G[ielectron]);
    // G1_e = inv(I - b11*dt*B)*G1_e
    invert_implicit_terms(G1[ielectron], f, b11*dt_);
    solver_->fieldSolve(G1, f);
    if (pars_->dealias_kz) grad_par->dealias(f->phi);
  } 

  // stage 2
  // compute A1 = A(G1)
  explicit_terms(A1, G1, f, true);
  // compute B1 = B(G1)
  implicit_terms(B1, G1, f);
  // G1_i = G_i + a21*dt*A1_i
  for(int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron) { // electrons
      G1[is]->set_zero();
    } else {
      G1[is]->add_scaled(1., G[is], a21*dt_, A1[is]);
    }
  }
  // compute Phi1_i (with G1_e=0)
  solver_->fieldSolve(G1, f);         
  // G1_e = G_e + a21*dt*A1_e + b21*dt*B1_e
  G1[ielectron]->add_scaled(1., G[ielectron], a21*dt_, A1[ielectron], b21*dt_, B1[ielectron]);
  // G1_e = inv(I - b22*dt*B)*G1_e
  invert_implicit_terms(G1[ielectron], f, b22*dt_);
  solver_->fieldSolve(G1, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  // compute A2 = A(G1)
  explicit_terms(A2, G1, f, false);
  // compute B2 = B(G1)
  implicit_terms(B2, G1, f);
  // G1_i = G_i + a31*A1_i + a32*A2_i + b31*dt*B1_i + b32*dt*B2_i
  for (int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron) { // electrons
      G1[is]->set_zero();
    } else {
      G1[is]->add_scaled(1., G[is], a31*dt_, A1[is], a32*dt_, A2[is], b31*dt_, B1[is], b32*dt_, B2[is]);
    }
  }
  // compute Phi_i (with G1_e=0)
  solver_->fieldSolve(G1, f);         
  // G1_e = G_e + a31*A1_e + a32*A2_e + b31*dt*B1_e + b32*dt*B2_e
  G1[ielectron]->add_scaled(1., G[ielectron], a31*dt_, A1[ielectron], a32*dt_, A2[ielectron], 
		            b31*dt_, B1[ielectron], b32*dt_, B2[ielectron]);
  // G1 = inv(I - b33*dt*B)*G1
  invert_implicit_terms(G1[ielectron], f, b33*dt_);
  solver_->fieldSolve(G1, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  // stage 4
  if(nstage>3) {
    // compute A3 = A(G1)
    explicit_terms(A3, G1, f, false);
    // compute B3 = B(G1)
    implicit_terms(B3, G1, f);
    // G1_i = G_i + a41*A1_i + a42*A2_i + a43*A3_i + b41*dt*B1_i + b42*dt*B2_i + b43*dt*B3_i
    for (int is=0; is<grids_->Nspecies; is++) {
      if(is == ielectron) { // electrons
        G1[is]->set_zero();
      } else {
	G1[is]->add_scaled(1., G[is], a41*dt_, A1[is], b41*dt_, B1[is]);
	G1[is]->add_scaled(1., G1[is], a42*dt_, A2[is], b42*dt_, B2[is]);
	G1[is]->add_scaled(1., G1[is], a43*dt_, A3[is], b43*dt_, B3[is]);
      }
    }
    // compute Phi_i (with G1_e=0)
    solver_->fieldSolve(G1, f);         
    // G1_e = G_e + a41*A1_e + a42*A2_e + b41*dt*B1_e + b42*dt*B2_e
    G1[ielectron]->add_scaled(1., G[ielectron], a41*dt_, A1[ielectron], a42*dt_, A2[ielectron], 
          	            b41*dt_, B1[ielectron], b42*dt_, B2[ielectron]);
    G1[ielectron]->add_scaled(1., G1[ielectron], a43*dt_, A3[ielectron], b43*dt_, B3[ielectron]);
    // G1 = inv(I - b44*dt*B)*G1
    invert_implicit_terms(G1[ielectron], f, b44*dt_);
    solver_->fieldSolve(G1, f);         
    if (pars_->dealias_kz) grad_par->dealias(f->phi);
    // compute A4 = A(G1)
    explicit_terms(A4, G1, f, false);
    // compute B4 = B(G1)
    implicit_terms(B4, G1, f);
  }
  // combine stage
  // G = G + w1*A1 + w2*A2 + w3*A3 + w4*A4 + w1*B1 + w2*B2 + w3*B3 + w4*B4
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is], w4*dt_, A4[is]); 
    G[is]->add_scaled(1., G[is], w1*dt_, B1[is], w2*dt_, B2[is], w3*dt_, B3[is], w4*dt_, B4[is]); 
  } 

  solver_->fieldSolve(G, f);        
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  for(int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}
