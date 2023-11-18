#include "timestepper.h"

// ======= IMEX SSP-RK3 + DIRK =======
IMEX_SSPRK3_DIRK::IMEX_SSPRK3_DIRK(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), ielectron(-1)
{
  
  // new objects for temporaries
  A0 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  A1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  A2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B0 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  B2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies); 
  for(int is=0; is<grids_->Nspecies; is++) {
    A0[is] = new MomentsG (pars_, grids_, is);
    A1[is] = new MomentsG (pars_, grids_, is);
    A2[is] = new MomentsG (pars_, grids_, is);
    B0[is] = new MomentsG (pars_, grids_, is);
    B1[is] = new MomentsG (pars_, grids_, is);
    B2[is] = new MomentsG (pars_, grids_, is);
    G1[is] = new MomentsG (pars_, grids_, is);
    
    // get species index of electrons
    if(pars_->species_h[is].type == 1) {
      ielectron = is;
      vte = A0[ielectron]->species->vt;
      zte = A0[ielectron]->species->zt;
    }
  }

  f1 = new Fields(pars_, grids_);

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
  else if (pars_->boundary_option_periodic) {
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }
  
  int nxkyz = grids_->NxNycNz;
  int nbx = min(32, nxkyz);
  int ngx = 1 + (nxkyz-1)/nbx;
  dB = dim3(nbx, 1, 1);
  dG = dim3(ngx, 1, 1);
  
}

IMEX_SSPRK3_DIRK::~IMEX_SSPRK3_DIRK()
{
  if (A0)    delete A0; 
  if (A1)    delete A1; 
  if (A2)    delete A2; 
  if (B0)    delete B0; 
  if (B1)    delete B1; 
  if (B2)    delete B2; 
  if (G1)    delete G1; 
  if (grad_par) delete grad_par;
}

void IMEX_SSPRK3_DIRK::explicit_terms(MomentsG** A, MomentsG** G, Fields* f, bool setdt)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    A[is]->set_zero();
    if(is == ielectron) {
      // compute explicit part of electron linear rhs
      linear_->rhs_nonstreaming(G[is], f, A[is], dt_);
    } else {
      // handle entire ion linear rhs explicitly
      linear_->rhs(G[is], f, A[is], dt_);
    }
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, A[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
  }
  // compute nonlinear terms explicitly for all species
}

void IMEX_SSPRK3_DIRK::implicit_terms(MomentsG** B, MomentsG** G, Fields* f)
{
  for (int is=0; is<grids_->Nspecies; is++) {
    B[is]->set_zero();
    if(is == ielectron) { // electrons
      // compute implicit part of electron linear rhs
      linear_->rhs_streaming(G[is], f, B[is], dt_);
    }
  }
}

void IMEX_SSPRK3_DIRK::invert_implicit_terms(MomentsG** G1, double rdt)
{
  // TBI
}

void IMEX_SSPRK3_DIRK::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }

  double q_ = 0.;
  double r_ = 1.;
  double s_ = 1./6.;
  double t_ = -1./3.;
  double u_ = 2./3.;
  
  // stage 1
  // compute A0 = F_explicit(G)
  explicit_terms(A0, G, f, true);
  // compute B0 = F_implicit(G)
  implicit_terms(B0, G, f);
  // G1_i = G_i + dt*A0_i + q_*dt*B0_i
  for(int is=0; is<grids_->Nspecies; is++) {
    if(is == ielectron) { // electrons
      G1[is]->set_zero();
    } else {
      G1[is]->add_scaled(1., G[is], dt_, A0[is], q_*dt_, B0[is]);
    }
  }
  // compute Phi_i
  solver_->fieldSolve(G1, f);         
  // G1_e = G_e + dt*A0_e + q_*dt*B0_e
  G1[ielectron]->add_scaled(1., G[ielectron], dt_, A0[ielectron], q_*dt_, B0[ielectron]);
  // compute grad_par Phi_i
  grad_par->dz(f->phi, f1->phi);
  // G_01e = G01_e - r_*dt*vte*zte*grad_par(Phi_i)
  add_scaled_singlemom_kernel <<<dG, dB>>> (G1[ielectron]->G(0,1), 1., G1[ielectron]->G(0,1), -r_*dt_*vte*zte, f1->phi);

  // G1 = inv(I - r_*dt*F_implicit)*G1
  invert_implicit_terms(G1, r_*dt_);
  solver_->fieldSolve(G1, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 2
  // compute A1 = F_explicit(G1)
  explicit_terms(A1, G1, f, false);
  // compute B1 = F_implicit(G1)
  implicit_terms(B1, G1, f);
  // G1 = G + dt/4*A0 + dt/4*A1 + s_*dt*B0 + t_*dt*B1
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is] -> add_scaled(1., G[is], dt_/4., A0[is], dt_/4., A1[is], s_*dt_, B0[is], t_*dt_, B1[is]);
 
  }

  // G1 = inv(I - u_*dt*F_implicit)*G1
  invert_implicit_terms(G1, u_*dt_);
  solver_->fieldSolve(G1, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  // compute A2 = F_explicit(G1)
  explicit_terms(A2, G1, f, false);
  // compute B2 = F_implicit(G1)
  implicit_terms(B2, G1, f);
  // G = G + dt/6*A0 + dt/6*A1 + 2*dt/3*A2 + dt/6*B0 + dt/6*B1 + 2*dt/3*B2
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> add_scaled(1., G[is], dt_/6., A0[is], dt_/6., A1[is], dt_/3., A2[is]);
    G[is] -> add_scaled(1., G[is], dt_/6., B0[is], dt_/6., B1[is], dt_/3., B2[is]); 
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
