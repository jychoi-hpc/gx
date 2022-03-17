#include "timestepper.h"
// #include "get_error.h"

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr)
{
  // new objects for temporaries
  GRhs  = new MomentsG (pars, grids);
  G1    = new MomentsG (pars, grids);
}

RungeKutta2::~RungeKutta2()
{
  if (GRhs)  delete GRhs;
  if (G1)    delete G1;
}

// ======== rk2  ==============
void RungeKutta2::EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* G, MomentsG* GRhs,
			    Fields* f, double adt, bool setdt)
{
  linear_->rhs(G0, f, GRhs); 
  
  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G0, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, adt*dt_, GRhs); 

}

void RungeKutta2::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G -> update_tprim(*t); 
  G1-> update_tprim(*t); 
  // end updates

  EulerStep (G1, G, G, GRhs, f, 0.5, true);    solver_->fieldSolve(G1, f);
  EulerStep (G, G1, G, GRhs, f, 1.0, false);   

  if (forcing_ != nullptr) forcing_->stir(G);  
   
  solver_->fieldSolve(G, f);

  *t += dt_;
}

