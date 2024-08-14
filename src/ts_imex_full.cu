#include "timestepper.h"
#include <stdio.h>
// ======= 3-stage addivte RK IMEX methods =======
IMEX_3stage_Full::IMEX_3stage_Full(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv, const float* kperp2) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
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
  Gc = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  Gr = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G0 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  phi_l = (cuComplex**) malloc(sizeof(void*)*grids_->Nspecies);

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
    G0[is] = new MomentsG (pars_, grids_, is_glob);
    checkCuda(cudaMalloc((void**) &phi_l[is], sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));
  
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
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);  
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
    linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
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
    linear_->rhs_streaming(G[is], f, B[is], dt_);
  }
}

void IMEX_3stage_Full::invert_implicit_terms(MomentsG** G1, MomentsG** Gc, MomentsG** Gr, MomentsG** G0, Fields *f, cuComplex** phi_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron)
{
  int max_iter = pars_->implicit_max_iter;
  for(int is = 0; is < grids_->Nspecies; is++){
    grad_par->zft(G1[is]);
  }

  for(int count = 0; count < max_iter; count++){
    if(count == 0){
      tridiag_streaming_periodic_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, 0, false);
      tridiag_streaming_periodic_full<<<dG, dB>>>(Gc[0]->G(), Gc[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(Gc[0]->species), *(Gc[ielectron]->species), sdt, gradpar_, 1, false);
//      tridiag_streaming_periodic_full<<<dG, dB>>>(Gr[0]->G(), Gr[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(Gr[0]->species), *(Gr[ielectron]->species), sdt, gradpar_, 2, false);

      sherman_morrison_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0]->G(), Gc[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), solver_->get_max_qneutFacPhi_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_);

      for(int is = 0; is < grids_->Nspecies; is++){
        grad_par->zft_inverse(G1[is]);
      }
    }
    else{
      solver_->fieldSolve(G1, f);
//      grad_par->zft(f->phi,f->phi);
      for(int is = 0; is < grids_->Nspecies; is++){
	apply_flr_phi<<<dG, dB>>>(phi_l[is], f->phi, kperp2_, *(G1[is]->species));
	grad_par->zft_nmoms(phi_l[is], phi_l[is], grids_->Nl);
        Gr[is]->copyFrom(G1[is]);
	G1[is]->copyFrom(G0[is]);
	grad_par->zft(G1[is]);
	grad_par->zft(Gr[is]);
      }

      tridiag_streaming_periodic_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_, 0, true);
      tridiag_streaming_periodic_full<<<dG, dB>>>(Gc[0]->G(), Gc[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(Gc[0]->species), *(Gc[ielectron]->species), sdt, gradpar_, 1, false);
//      tridiag_streaming_periodic_full<<<dG, dB>>>(Gr[0]->G(), Gr[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), phi_l[0], phi_l[ielectron], grids_->kz, solver_->get_max_qneutFacPhi_inv(), *(Gr[0]->species), *(Gr[ielectron]->species), sdt, gradpar_, 2, false);

      sherman_morrison_full<<<dG, dB>>>(G1[0]->G(), G1[ielectron]->G(), Gc[0]->G(), Gc[ielectron]->G(), Gr[0]->G(), Gr[ielectron]->G(), solver_->get_max_qneutFacPhi_inv(), grids_->kz, *(G1[0]->species), *(G1[ielectron]->species), sdt, gradpar_);

      for(int is = 0; is < grids_->Nspecies; is++){
	grad_par->zft_inverse(G1[is]);
      }

    }
  }

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
  double a21, a31, a32, w1, w2, w3;
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
  }
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
//    invert_implicit_terms(G1[ielectron], f, p_*dt_,gradpar_);
    invert_implicit_terms(G1, Gc, Gr, G0, f, phi_l, p_*dt_,gradpar_, bmagInv_, ielectron);

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
  invert_implicit_terms(G1, Gc, Gr, G0, f, phi_l, r_*dt_,gradpar_, bmagInv_, ielectron);
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
  invert_implicit_terms(G1, Gc, Gr, G0, f, phi_l, u_*dt_,gradpar_,bmagInv_, ielectron);
    solver_->fieldSolve(G1, f);          
  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  // combine stage
  // compute A3 = A(G1)
  explicit_terms(A3, G1, f, false);
  // compute B3 = B(G1)
  implicit_terms(B3, G1, f);
  // G = G + w1*A1 + w2*A2 + w3*A3 + w1*B1 + w2*B2 + w3*B3
  for (int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G[is], w1*dt_, A1[is], w2*dt_, A2[is], w3*dt_, A3[is]); 
    G[is]->add_scaled(1., G[is], w1*dt_, B1[is], w2*dt_, B2[is], w3*dt_, B3[is]); 
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
