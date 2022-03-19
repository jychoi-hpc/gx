#include "timestepper.h"
// #include "get_error.h"

K2::K2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), 
  G_q1(nullptr), G_q2(nullptr)
{
  stages_ = pars_->stages;
  sm1inv = (double) 1./((double) stages_-1);
  sinv   = (double) 1./((double) stages_);
  
  // new objects for temporaries
  G_q1  = new MomentsG (pars_, grids_);
  G_q2  = new MomentsG (pars_, grids_);
}

K2::~K2()
{
  if (G_q1)  delete G_q1;
  if (G_q2)  delete G_q2;
}

void K2::EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, bool setdt)
{  
  linear_->rhs(G_q1, f, GRhs);

  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G_q1, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }

  G_q1->add_scaled(1., G_q1, dt_*sm1inv, GRhs);
  solver_->fieldSolve(G_q1, f);    
}

void K2::FinalStep(MomentsG* G_q1, MomentsG* G_q2, MomentsG* GRhs, Fields* f)
{  
  linear_->rhs(G_q1, f, GRhs);
  if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1, f, GRhs);

  double sm1 = (double) stages_ - 1;  
  GRhs->add_scaled(sm1*sinv, G_q1, sinv, G_q2, dt_*sinv, GRhs);
  // no field solve required here
}

void K2::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G   -> update_tprim(*t); 
  G_q1-> update_tprim(*t); 
  G_q2-> update_tprim(*t); 
  // end updates

  G_q1->copyFrom(G);
  G_q2->copyFrom(G);

  bool setdt = true;
  
  for(int i=1; i<stages_; i++) {
    EulerStep(G_q1, G, f, setdt);
    setdt = false;
  }

  FinalStep(G_q1, G_q2, G, f); // returns G 
  if (forcing_ != nullptr) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
}

