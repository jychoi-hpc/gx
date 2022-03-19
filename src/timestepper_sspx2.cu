#include "timestepper.h"

// ======= SSPx2 =======
SSPx2::SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr)
{
  // new objects for temporaries
  GRhs  = new MomentsG (pars, grids);
  G1    = new MomentsG (pars, grids);
  G2    = new MomentsG (pars, grids);
}

SSPx2::~SSPx2()
{
  if (GRhs)  delete GRhs;
  if (G1)    delete G1; 
  if (G2)    delete G2; 
}

// ======== SSPx2  ==============
void SSPx2::EulerStep(MomentsG* G1, MomentsG* G, MomentsG* GRhs, Fields* f, bool setdt)
{
  linear_->rhs(G, f, GRhs);

  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, dt_/sqrt(2.), GRhs);

}

void SSPx2::advance(double *t, MomentsG* G, Fields* f)
{

  // update the gradients if they are evolving
  G -> update_tprim(*t); 
  G1-> update_tprim(*t); 
  // end updates
  
  EulerStep (G1, G, GRhs, f, true); 
  solver_->fieldSolve(G1, f);

  EulerStep (G2, G1, GRhs, f, false);

  G->add_scaled(2.-sqrt(2.), G, sqrt(2.)-2., G1, 1., G2);
  
  if (forcing_ != nullptr) forcing_->stir(G);  

  solver_->fieldSolve(G, f);

  *t += dt_;
}
