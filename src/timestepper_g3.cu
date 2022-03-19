#include "timestepper.h"
// #include "get_error.h"

G3::G3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  G_u1(nullptr), G_u2(nullptr)
{
  // new objects for temporaries
  G_u1  = new MomentsG (pars_, grids_);
  G_u2  = new MomentsG (pars_, grids_);
}

G3::~G3()
{
  if (G_u1)  delete G_u1;
  if (G_u2)  delete G_u2;
}

void G3::EulerStep(MomentsG* G_u, MomentsG* GRhs, Fields* f, bool setdt)
{
  linear_->rhs(G_u, f, GRhs);

  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G_u, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }

  G_u->add_scaled(1., G_u, dt_, GRhs);
  solver_->fieldSolve(G_u, f);    
}

void G3::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G   -> update_tprim(*t); 
  G_u1-> update_tprim(*t); 
  //  G_u2-> update_tprim(*t); 
  // end updates

  G_u1->copyFrom(G);
  G_u2->copyFrom(G);

  float onethird = 1./3.;
  float twothirds = 2./3.;  
  bool setdt = true;
  
  EulerStep(G_u1, G, f, setdt);
  setdt = false;

  EulerStep(G_u1, G, f, setdt);

  G_u1->add_scaled(0.75, G_u2, 0.25, G_u1);
  solver_->fieldSolve(G_u1, f);
  
  EulerStep(G_u1, G, f, setdt);
  G->add_scaled(onethird, G_u2, twothirds, G_u1);

  if (forcing_ != nullptr) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  
}
