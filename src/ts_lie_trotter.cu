#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
Lie_Trotter::Lie_Trotter(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Cusolve *cusolve, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  cusolve_(cusolve), forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), bmagInv_(bmagInv)
{
  
  // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gc = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gr = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    A1[is] = new MomentsG (pars_, grids_, is_glob);
    A2[is] = new MomentsG (pars_, grids_, is_glob);
    A3[is] = new MomentsG (pars_, grids_, is_glob);
    B1[is] = new MomentsG (pars_, grids_, is_glob);
    B2[is] = new MomentsG (pars_, grids_, is_glob);
    B3[is] = new MomentsG (pars_, grids_, is_glob);
    G1[is] = new MomentsG (pars_, grids_, is_glob);
    Gc[is] = new MomentsG (pars_, grids_, is_glob);
    Gr[is] = new MomentsG (pars_, grids_, is_glob);
    
    // get species index of electrons
    if(pars_->species_h[is].type == 1) {
      ielectron = is;
      vte = A1[ielectron]->species->vt;
    }
  }

  bounce_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  res = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  size_t brhs_size = sizeof(cuComplex)*pars_->nm_in*pars_->nl_in*grids_->Nyc*grids_->Nx;
  for (int j = 0; j < grids_->Nz; j++){
    checkCuda(cudaMalloc((void**) &bounce_rhs[j], brhs_size));
    checkCuda(cudaMalloc((void**) &res[j], brhs_size));
  }

//  checkCuda(cudaMalloc((void**) &bounce_rhs, brhs_size));
//  checkCuda(cudaMalloc((void**) &res, brhs_size));

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
  int nn4 = pars_->nm_in * pars_->nl_in; int nt4 = min(nn4, 4); int nb4 = 1 + (nn4-1)/nt4;

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_b = dim3(nt1, nt2, nt4);
  dB_b = dim3(nb1, nb2, nb4);
 

  flip = true;
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
  if (grad_par) delete grad_par;
}

void Lie_Trotter::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron) {
//      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
      linear_->rhs_nonstreaming_nonbounce(G[is], f, A[is], dt_);
//      linear_->rhs(G[is],f,A[is],dt_);
    } else {
      linear_->rhs(G[is], f, A[is], dt_);
      }
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void Lie_Trotter::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron) { // electrons
      // compute implicit part of electron linear rhs
//      linear_->rhs_streaming(G[is], f, B[is], dt_);
      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);


    }
  }
}

void Lie_Trotter::invert_implicit_terms(MomentsG** G1, MomentsG* Gc, MomentsG** Gr, Fields *f, double sdt,const float gradpar_, const float* bmagInv_, int ielectron, bool flip)
{
/*  if(flip){
    invert_bounce_terms(G1[ielectron], 0);
  }*/
  double sdtvt = sdt*vte;
  // tridiag from numerical recipes
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
  } else if (pars_->boundary_option_periodic) {
      // FFT Phi_i
      grad_par->zft(f->phi, f->phi);
      grad_par->zft(f->apar, f->apar);

      // FFT G_e
      grad_par->zft(G1[ielectron]);

      int max_iter = pars_->implicit_max_iter;
      double omega = pars_->implicit_omega;
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

      int max_iter = pars_->implicit_max_iter;
      double omega = pars_->implicit_omega;
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
      int max_iter = pars_->implicit_max_iter;
      double omega = pars_->implicit_omega;
      for (int count = 0; count < max_iter; count++){
	if (count ==0){
          Gr[ielectron]->copyFrom(G1[ielectron]);
          checkCudaErrors(cudaGetLastError());
	  if(pars_->fapar > 0. || pars_->fbpar > 0.){
            checkCudaErrors(cudaGetLastError());
	    grad_par->zft_streaming_invert_em(G1[ielectron], Gr[ielectron],f->phi,f->apar,f->bpar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_qneutFacBpar_inv(),solver_->get_max_ampereParFac_inv(),solver_->get_max_amperePerpFacPhi_inv(),solver_->get_max_amperePerpFacBpar_inv(),sdtvt, gradpar_, false);

	  }
	  else{
            grad_par->zft_streaming_invert(G1[ielectron], Gr[ielectron], f->phi,solver_->getQneutDenom(),solver_->get_max_qneutFacPhi_inv(),sdtvt, gradpar_, false); 
	  }
	}
	else{
	  grad_par->zft_inverse(G1[ielectron]); //Calculate full potential
          Gr[ielectron]->copyFrom(G1[ielectron]);
          solver_->fieldSolve(G1, f);

          G1[ielectron]->copyFrom(Gc); //I think for iteration scheme, need original G1

          if(pars_->fapar > 0. || pars_->fbpar > 0.){
            grad_par->zft_streaming_invert_em(G1[ielectron], Gr[ielectron],f->phi,f->apar,f->bpar,solver_->get_max_qneutFacPhi_inv(),solver_->get_max_qneutFacBpar_inv(),solver_->get_max_ampereParFac_inv(),solver_->get_max_amperePerpFacPhi_inv(),solver_->get_max_amperePerpFacBpar_inv(),sdtvt, gradpar_, true);
          }
          else{
            grad_par->zft_streaming_invert(G1[ielectron], Gr[ielectron], f->phi,solver_->getQneutDenom(),solver_->get_max_qneutFacPhi_inv(),sdtvt, gradpar_, true);
          }

          G1[ielectron]->add_scaled(omega,G1[ielectron],(1.-omega),Gr[ielectron]);
	}
      }
    }
  grad_par->zft_inverse(G1[ielectron]);
  invert_bounce_terms(G1[ielectron], 0);

/*  if(!flip){
    invert_bounce_terms(G1[ielectron], 0);
  }*/

}

void Lie_Trotter::invert_bounce_terms(MomentsG* G, int stage){
/*  for (int iz = 0; iz < grids_->Nz; iz++){
    copy_brhs_from_g<<<dG_b, dB_b>>>(bounce_rhs, G->G(), iz);
//    copy_brhs_from_g<<<dG_b, dB_b>>>(res, G->G(), iz);
    cusolve_->invert(bounce_rhs, res, iz);
    copy_g_from_brhs<<<dG_b, dB_b>>>(G->G(),bounce_rhs, iz);
    //check_residual<<<dG_b, dB_b>>>(res);
  }*/

  for (int iz = 0; iz < grids_->Nz; iz++){
    copy_brhs_from_g<<<dG_b, dB_b>>>(bounce_rhs[iz],G->G(), iz);
    copy_brhs_from_g<<<dG_b, dB_b>>>(res[iz], G->G(), iz);
    cusolve_->invert(bounce_rhs[iz], res[iz], iz);
    copy_g_from_brhs<<<dG_b, dB_b>>>(G->G(),bounce_rhs[iz], iz);
 //   check_residual<<<dG_b, dB_b>>>(res[iz]);

  }


}
void Lie_Trotter::ssprk3(MomentsG** A1, MomentsG** A2, MomentsG** A3, MomentsG** G, MomentsG** G1, Fields* f, bool setdt){
  explicit_terms(A1, G, f, false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], dt_, A1[is]);
  }
  solver_->fieldSolve(G1,f);

  explicit_terms(A2,G1,f,false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(1., G[is], 1./4.*dt_, A1[is], 1./4.*dt_, A2[is]);
  }
  solver_->fieldSolve(G1,f);

  explicit_terms(A3,G1,f,false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], 1./6.*dt_, A1[is], 1./6.*dt_, A2[is], 2./3.*dt_, A3[is]);
  }
  solver_->fieldSolve(G,f);

/*  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], dt_, A1[is]);
  }*/


}

void Lie_Trotter::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }

  if(flip){
    checkCudaErrors(cudaGetLastError()); 
    ssprk3(A1, A2, A3, G, G1, f, false); 

    for(int is=0; is<grids_->Nspecies; is++) {
      if (is == ielectron){
        G1[ielectron]->set_zero();
      }
      else{
        G1[is]->copyFrom(G[is]);
      } 
    }

    solver_->fieldSolve(G1, f);         
    Gc[ielectron]->copyFrom(G[ielectron]);
    Gr[ielectron]->copyFrom(G[ielectron]);
    invert_implicit_terms(G, Gc[ielectron], Gr, f, 1.*dt_,gradpar_, bmagInv_, ielectron, flip);
    solver_->fieldSolve(G, f);
  }
  else{
    for(int is=0; is<grids_->Nspecies; is++) {
      if (is == ielectron){
        G1[ielectron]->set_zero();
      }
      else{
        G1[is]->copyFrom(G[is]);
      } 
    }

    solver_->fieldSolve(G1, f);         
    Gc[ielectron]->copyFrom(G[ielectron]);
    Gr[ielectron]->copyFrom(G[ielectron]);
    invert_implicit_terms(G, Gc[ielectron], Gr, f, dt_,gradpar_, bmagInv_, ielectron, flip);
    solver_->fieldSolve(G, f);        
    
    checkCudaErrors(cudaGetLastError()); 
    ssprk3(A1, A2, A3, G, G1, f, false); 
    solver_->fieldSolve(G,f);

  }

/*  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  for (int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);*/
  *t += dt_;
  checkCudaErrors(cudaGetLastError());
  //flip = !flip;
}

