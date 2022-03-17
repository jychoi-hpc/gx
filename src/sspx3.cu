#include "timestepper.h"
// #include "get_error.h"

/*
This method was set up by Greg Hammett (and a student, see below), 
with a minor correction from Dorland.

The derivation of the method is laid out in the file SSPX3-4_2016.12.01.pdf 
in the docs/files directory of this repo, with a correction needed for the 
value of w3 on page 5. This required correction appears in the specification 
of w3 for this method, which appears in ../include/timestepper.h. 

The notes in the PDF above are from math student Federico Pasqualotto. 
*/

// ======= SSPx3 =======
SSPx3::SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr), G3(nullptr)
{
  
  // new objects for temporaries
  GRhs  = new MomentsG (pars_, grids_);
  G1    = new MomentsG (pars_, grids_);
  G2    = new MomentsG (pars_, grids_);
  G3    = new MomentsG (pars_, grids_);

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

SSPx3::~SSPx3()
{
  if (GRhs)  delete GRhs;
  if (G1)    delete G1; 
  if (G2)    delete G2; 
  if (G3)    delete G3; 
  if (grad_par) delete grad_par;
}

// ======== SSPx3  ==============
void SSPx3::EulerStep(MomentsG* G1, MomentsG* G, MomentsG* GRhs, Fields* f, bool setdt)
{
  linear_->rhs(G, f, GRhs);  if (pars_->dealias_kz) grad_par->dealias(GRhs);

  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }
  if (pars_->dealias_kz) grad_par->dealias(GRhs);

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, adt*dt_, GRhs);
}

void SSPx3::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G -> update_tprim(*t); 
  G1-> update_tprim(*t); 
  G2-> update_tprim(*t); 
  // end updates
  
  EulerStep (G1, G , GRhs, f, true);  
  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);
  EulerStep (G2, G1, GRhs, f, false); 

  G2->add_scaled((1.-w1), G, (w1-1.), G1, 1., G2);
  solver_->fieldSolve(G2, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  EulerStep (G3, G2, GRhs, f, false);

  G->add_scaled((1.-w2-w3), G, w3, G1, (w2-1.), G2, 1., G3);
  
  if (forcing_ != nullptr) forcing_->stir(G);  
  G->mask();
  solver_->fieldSolve(G, f);          if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}

