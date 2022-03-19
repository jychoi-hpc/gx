#include "timestepper.h"
// #include "get_error.h"

// ============= RK4 =============
RungeKutta4::RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  GStar(nullptr), GRhs(nullptr), G_q1(nullptr), G_q2(nullptr)
{
  GStar = new MomentsG (pars, grids);
  GRhs  = new MomentsG (pars, grids);
  G_q1  = new MomentsG (pars, grids);
  G_q2  = new MomentsG (pars, grids);
}

RungeKutta4::~RungeKutta4()
{
  if (GStar) delete GStar;
  if (GRhs) delete GRhs;
  if (G_q1) delete G_q1;
  if (G_q2) delete G_q2;
}

// ======== rk4  ==============

void RungeKutta4::partial(MomentsG* G, MomentsG* Gt, Fields *f, MomentsG* Rhs, MomentsG *Gnew, double adt, bool setdt)
{
  linear_->rhs(Gt, f, Rhs);
  if (nonlinear_ != nullptr) {
    nonlinear_->nlps (Gt, f, Rhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }

  if (pars_->eqfix) Gnew->copyFrom(G);
  
  Gnew->add_scaled(1., G, adt*dt_, Rhs);
  solver_->fieldSolve(Gnew, f);
}

void RungeKutta4::advance(double *t, MomentsG* G, Fields* f)
{

  // update the gradients if they are evolving
  G   -> update_tprim(*t); 
  G_q1-> update_tprim(*t); 
  G_q2-> update_tprim(*t); 
  // end updates

  partial(G, G,    f, GRhs,  G_q1, 0.5, true);
  partial(G, G_q1, f, GStar, G_q2, 0.5, false);

  // Do a partial accumulation of final update to save memory
  GRhs->add_scaled(dt_/6., GRhs, dt_/3., GStar);

  partial(G, G_q2, f, GStar, G_q1, 1., false);

  // This update is just to improve readability
  GRhs->add_scaled(1., GRhs, dt_/3., GStar);
  
  linear_->rhs(G_q1, f, GStar);
  if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1, f, GStar);     
  
  G->add_scaled(1., G, 1., GRhs, dt_/6., GStar);

  /*
  partial(G, G_q2, f, G_q1, GStar, 1., false);

  linear_->rhs(GStar, f, G_q2);               
  if(nonlinear_ != nullptr) nonlinear_->nlps(GStar, f, G_q2);     
  
  G->add_scaled(1., G, 1., GRhs, dt_/3., G_q1, dt_/6., G_q2);
  */
  
  if (forcing_ != nullptr) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
}

