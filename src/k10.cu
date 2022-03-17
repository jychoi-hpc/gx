#include "timestepper.h"
// #include "get_error.h"

// This method is described in chapter 6 of David Ketcheson's thesis:
// https://www.davidketcheson.info/assets/papers/ketchesonphdthesis.pdf

// The option to dealias in kz presently requires extra FFTs and only works for
// systems without magnetic shear. It has been tested with slab ETG turbulence and
// works very well. It should not be assumed that this way of dealiasing in kz
// will work in any more general circumstances. 

// ============= K10,4 ============
Ketcheson10::Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), G_q1(nullptr), G_q2(nullptr)
{
  // new objects for temporaries
  G_q1  = new MomentsG (pars_, grids_);
  G_q2  = new MomentsG (pars_, grids_);

  if (pars_->local_limit)                     { grad_par = new GradParallelLocal(grids_);
  } else if (pars_->boundary_option_periodic) { grad_par = new GradParallelPeriodic(grids_);
  } else {                                      grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
}

Ketcheson10::~Ketcheson10()
{
  if (G_q1)  delete G_q1;
  if (G_q2)  delete G_q2;
  if (grad_par) delete grad_par;
}

void Ketcheson10::EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, bool setdt)
{
  linear_->rhs(G_q1, f, GRhs);   if (pars_->dealias_kz) grad_par->dealias(GRhs);
  
  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G_q1, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }
  if (pars_->dealias_kz) grad_par->dealias(GRhs);

  G_q1->add_scaled(1., G_q1, dt_/6., GRhs);
  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
}

void Ketcheson10::advance(double *t, MomentsG* G, Fields* f)
{
  bool setdt = true;

  // update the gradients if they are evolving
  G    -> update_tprim(*t); 
  G_q1 -> update_tprim(*t); 
  G_q2 -> update_tprim(*t); 
  // end updates

  G_q1 -> copyFrom(G);
  G_q2 -> copyFrom(G);

  for(int i=1; i<6; i++) { EulerStep(G_q1, G, f, setdt);    setdt = false;}

  G_q2 -> add_scaled(0.04, G_q2, 0.36, G_q1);
  G_q1 -> add_scaled(15, G_q2, -5, G_q1);

  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  
  for(int i=6; i<10; i++) EulerStep(G_q1, G, f, setdt);
  
  linear_->rhs(G_q1, f, G);
  if (pars_->dealias_kz) grad_par->dealias(G);
  
  if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1, f, G);
  if (pars_->dealias_kz) grad_par->dealias(G);
  
  G->add_scaled(1., G_q2, 0.6, G_q1, 0.1*dt_, G);
  
  if (forcing_ != nullptr) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  *t += dt_;
}

