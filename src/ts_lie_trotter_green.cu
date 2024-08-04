#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
Lie_Trotter_Green::Lie_Trotter_Green(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Green *green, Forcing *forcing, double dt_in, const float gradpar, const float* kperp2) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  green_(green), forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), kperp2_(kperp2)
{
  
  // new objects for temporaries
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gh = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    A1[is] = new MomentsG (pars_, grids_, is_glob);
    A2[is] = new MomentsG (pars_, grids_, is_glob);
    A3[is] = new MomentsG (pars_, grids_, is_glob);
    B1[is] = new MomentsG (pars_, grids_, is_glob);
    B2[is] = new MomentsG (pars_, grids_, is_glob);
    B3[is] = new MomentsG (pars_, grids_, is_glob);
    G1[is] = new MomentsG (pars_, grids_, is_glob);
    Gh[is] = new MomentsG (pars_, grids_, is_glob);
   
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
  checkCuda(cudaMalloc((void**) &phi_i, sizeof(cuComplex)*grids_->NxNycNz));

  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  int nn4 = pars_->nm_in * pars_->nl_in; int nt4 = min(nn4, 4); int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nz; int nt5 = min(nn5, 4); int nb5 = 1 + (nn5-1)/nt5;
  int nn6 = grids_->Nl; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_b = dim3(nt1, nt2, nt4);
  dB_b = dim3(nb1, nb2, nb4);
  
  dG_l = dim3(nt1, nt2, nt5);
  dB_l = dim3(nb1, nb2, nb5);


  flip = false;
}

Lie_Trotter_Green::~Lie_Trotter_Green()
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

void Lie_Trotter_Green::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
//    linear_->rhs(G[is],f,A[is],dt_);
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void Lie_Trotter_Green::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    linear_->rhs_streaming(G[is], f, B[is], dt_);
  }
}

void Lie_Trotter_Green::invert_implicit_terms(MomentsG** G, MomentsG** G1, Fields *f, double sdt,const float gradpar_, const float* kperp2, int ielectron)
{
  // tridiag from numerical recipes
  if (pars_->boundary_option_periodic) {
      for(int is = 0; is<grids_->Nspecies; is++){
	grad_par->zft(G1[is]);
        compute_inhomogenous_sol<<<dG, dB>>>(G1[is]->G(), grids_->kz, *(G1[is]->species), sdt, gradpar_);
	grad_par->zft_inverse(G1[is]);
      }
      solver_->fieldSolve(G1,f);

      green_->invert(f->phi);

      for(int is = 0; is<grids_->Nspecies; is++){
	for(int il = 0; il < grids_->Nl; il++){
          apply_flr_phi_loop<<<dG_l, dB_l>>>(phi_i,f->phi,kperp2,*(G[is]->species), il);
          grad_par->zft(phi_i,phi_i);
	  grad_par->zft(f->phi,phi_i);
	  grad_par->zft(G[is]); 
          compute_full_sol_loop<<<dG_l, dB_l>>>(G[is]->G(), phi_i, grids_->kz, *(G[is]->species), sdt, gradpar_, il);
	  grad_par->zft_inverse(G[is]);
	}
      }
  } 
//  cublas_->invert_stream(G1[ielectron]->G(), 0);
}


void Lie_Trotter_Green::ssprk3(MomentsG** A1, MomentsG** A2, MomentsG** A3, MomentsG** G, MomentsG** G1, Fields* f, bool setdt){
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


}

void Lie_Trotter_Green::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }
  if(!flip){
    checkCudaErrors(cudaGetLastError()); 
    ssprk3(A1, A2, A3, G, G1, f, false); 

    for(int is=0; is<grids_->Nspecies; is++) {
      G1[is]->copyFrom(G[is]);
    }

    invert_implicit_terms(G, G1, f, 1.*dt_,gradpar_, kperp2_, ielectron);
    solver_->fieldSolve(G, f);
  }
  else{
    for(int is=0; is<grids_->Nspecies; is++) {
      G1[is]->copyFrom(G[is]);
    }
    invert_implicit_terms(G, G1, f, 1.*dt_,gradpar_, kperp2_, ielectron);
    solver_->fieldSolve(G, f);        
    
    ssprk3(A1, A2, A3, G, G1, f, false); 
    solver_->fieldSolve(G,f);


    checkCudaErrors(cudaGetLastError()); 

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
//  flip = !flip;
}

