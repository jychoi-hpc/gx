#include "timestepper.h"

// ======= IMEX SSP-RK3 + DIRK =======
IMEX_SSPRK3_DIRK::IMEX_SSPRK3_DIRK(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  
  // new objects for temporaries
  A0 = new MomentsG (pars_, grids_);
  A1 = new MomentsG (pars_, grids_);
  A2 = new MomentsG (pars_, grids_);
  B0 = new MomentsG (pars_, grids_);
  B1 = new MomentsG (pars_, grids_);
  B2 = new MomentsG (pars_, grids_);
  G1 = new MomentsG (pars_, grids_);

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
  else if (pars_->boundary_option_periodic) {
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
  
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

void IMEX_SSPRK3_DIRK::explicit_terms(MomentsG* G1, MomentsG* G, Fields* f, bool setdt)
{
  G1->set_zero();
  for (int s=0; s<grids_->Nspecies; s++) {
    if(pars_->species_h[s].type == 1) {
      // compute explicit part of electron linear rhs
      //linear_->rhs_explicit(G, f, G1, s);
    } else {
      // handle entire ion linear rhs explicitly
      //linear_->rhs(G, f, G1, s);
    }
  }

  // compute nonlinear terms explicitly for all species
  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G, f, G1);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }
}

void IMEX_SSPRK3_DIRK::implicit_terms(MomentsG* G1, MomentsG* G, Fields* f)
{
  // TBI
}

void IMEX_SSPRK3_DIRK::invert_implicit_terms(MomentsG* G1, double rdt)
{
  // TBI
}

void IMEX_SSPRK3_DIRK::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G -> update_tprim(*t); 
  G1-> update_tprim(*t); 
  // end updates

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
  // G1 = G + dt*A0 + q_*dt*B0
  G1->add_scaled(1., G, dt_, A0, q_*dt_, B0);
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
  G1->add_scaled(1., G, dt_/4., A0, dt_/4., A1, s_*dt_, B0, t_*dt_, B1);
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
  G->add_scaled(1., G, dt_/6., A0, dt_/6., A1, dt_/3., A2); 
  G->add_scaled(1., G, dt_/6., B0, dt_/6., B1, dt_/3., B2); 
  solver_->fieldSolve(G, f);        
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  if (forcing_ != nullptr) forcing_->stir(G);  
  G->mask();
  solver_->fieldSolve(G, f);         
  if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}
