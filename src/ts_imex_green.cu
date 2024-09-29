#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
IMEX_3stage_Green::IMEX_3stage_Green(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Green *green, Cublas_test *cublas, Forcing *forcing, double dt_in, const float gradpar, const float* kperp2) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars), green_(green), cublas_(cublas), forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1), gradpar_(gradpar), kperp2_(kperp2)
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

  if(pars_->nstages > 3){
    A4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
    B4 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  }
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

    if(pars_->nstages > 3){
      A4[is] = new MomentsG (pars_, grids_, is_glob);
      B4[is] = new MomentsG (pars_, grids_, is_glob);
    }
    // get species index of electrons
    if(pars_->species_h[is].type == 1) {
      ielectron = is;
      vte = A1[ielectron]->species->vt;
    }
  }
//  checkCuda(cudaMalloc((void**) &phi_i, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc((void**) &phi_i, sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));

  checkCuda(cudaMalloc((void**) &phi_copy, sizeof(cuComplex)*grids_->NxNycNz));


  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
//  else if (pars_->boundary_option_periodic) {
//   grad_par = new GradParallelPeriodic(grids_);
//  }
  else {
    printf("USING GRADPARALLELLINKED!!!\n");
    grad_par = new GradParallelLinked(pars_, grids_);
    printf("AFTER GRADPARALLELINKED!!!\n");
  }
  
  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  int nn4 = pars_->nm_in * pars_->nl_in; int nt4 = min(nn4, 4); int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nz;       int nt5 = min(nn5, 4);    int nb5 = 1 + (nn5-1)/nt5;

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_b = dim3(nt1, nt2, nt4);
  dB_b = dim3(nb1, nb2, nb4);

  dG_l = dim3(nt1, nt2, nt5);
  dB_l = dim3(nb1, nb2, nb5);


  a21 = pars_->a21; a31 = pars_->a31; a32 = pars_->a32; w1 = pars_->w1, w2 = pars_->w2; w3 = pars_->w3;
  p_ = pars_->p_; q_ = pars_->q_; r_ = pars_->r_; s_ = pars_->s_; t_ = pars_->t_; u_ = pars_->u_;

  nstages = pars_->nstages;
  a41 = pars_->a41, a42 = pars_->a42, a43 = pars_->a43, a44 = pars_->a44;
  w_ = pars_->w_, x_ = pars_->x_, y_ = pars_->y_, z_ = pars_->z_;
  w4 = pars_->w4;
  sdirk = pars_->sdirk;

  flip = true;
}

IMEX_3stage_Green::~IMEX_3stage_Green()
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

void IMEX_3stage_Green::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if (is == ielectron){
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
}

void IMEX_3stage_Green::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if (is == ielectron){
      linear_->rhs_streaming_bounce(G[is], f, B[is], dt_);
    }
    else{
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}

void IMEX_3stage_Green::invert_implicit_terms(MomentsG** G1, MomentsG** Gh, Fields *f, double sdt,const float gradpar_, const float* kperp2, int ielectron)
{
  // tridiag from numerical recipes
//  if (pars_->boundary_option_periodic) {
    if(true){
      for(int is = 0; is<grids_->Nspecies; is++){
	grad_par->zft(Gh[is]);
        compute_inhomogenous_sol<<<dG, dB>>>(Gh[is]->G(), grids_->kz, *(Gh[is]->species), sdt, gradpar_);
	grad_par->zft_inverse(Gh[is]);
      }
      solver_->fieldSolve(Gh,f);
      green_->invert(f->phi);
      for(int is = 0; is<grids_->Nspecies; is++){
	apply_flr_phi<<<dG, dB>>>(phi_i, f->phi, kperp2, *(G1[is]->species));
	grad_par->zft_nmoms(phi_i, phi_i, grids_->Nl);
	grad_par->zft(G1[is]);
	compute_full_sol<<<dG, dB>>>(G1[is]->G(), phi_i, grids_->kz, *(G1[is]->species), sdt, gradpar_);
	grad_par->zft_inverse(G1[is]);
      }
  } 
  cublas_->invert_stream(G1[ielectron]->G(), 0);
}

void IMEX_3stage_Green::advance(double *t, MomentsG** G, Fields* f)
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
  checkCudaErrors(cudaGetLastError()); 
  // stage 1
  for (int is=0; is<grids_->Nspecies; is++) {
    G1[is]->copyFrom(G[is]);
    Gh[is]->copyFrom(G1[is]);
  }
  if(p_!=0.) {
    // compute Phi1_i (with G1_e=0)
    solver_->fieldSolve(G1, f);
    G1[ielectron]->copyFrom(G[ielectron]);
    // G1_e = inv(I - p_*dt*B)*G1_e
    invert_implicit_terms(G1, Gh, f, p_*dt_,gradpar_, kperp2_, ielectron);
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
      G1[is]->add_scaled(1., G[is], a21*dt_, A1[is], q_*dt_, B1[is]);
      Gh[is]->copyFrom(G1[is]);
  }
  // G1_e = inv(I - r_*dt*B)*G1_e
  invert_implicit_terms(G1, Gh, f, r_*dt_,gradpar_, kperp2_, ielectron);
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
      Gh[is]->copyFrom(G1[is]);
  }
 
  // G1 = inv(I - u_*dt*B)*G1
  invert_implicit_terms(G1, Gh, f, u_*dt_,gradpar_,kperp2_, ielectron); 
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
      Gh[is]->copyFrom(G1[is]);
  }
 
  // G1 = inv(I - u_*dt*B)*G1
  invert_implicit_terms(G1, Gh, f, z_*dt_,gradpar_,kperp2_, ielectron); 
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
/*  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  for (int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);*/
  *t += dt_;
  checkCudaErrors(cudaGetLastError());
  flip = !flip;
}

