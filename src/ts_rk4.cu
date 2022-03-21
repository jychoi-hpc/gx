#include "timestepper.h"
// #include "get_error.h"

// ============= RK4 =============
RungeKutta4::RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  GStar(nullptr), GRhs(nullptr), G_q1(nullptr), G_q2(nullptr)
{
  GStar = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  GRhs = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    GStar[is] = new MomentsG (pars_, grids_, is);
    GRhs[is] = new MomentsG (pars_, grids_, is);
    G_q1[is] = new MomentsG (pars_, grids_, is);
    G_q2[is] = new MomentsG (pars_, grids_, is);
  }
}

RungeKutta4::~RungeKutta4()
{
  for(int is=0; is<grids_->Nspecies; is++) {
    if (GStar[is]) delete GStar[is];
    if (GRhs[is]) delete GRhs[is];
    if (G_q1[is]) delete G_q1[is];
    if (G_q2[is]) delete G_q2[is];
  }
  free(GStar);
  free(GRhs);
  free(G_q1);
  free(G_q2);
}

// ======== rk4  ==============

void RungeKutta4::partial(MomentsG** G, MomentsG** Gt, Fields *f, MomentsG** Rhs, MomentsG **Gnew, double adt, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    Rhs[is]->set_zero();
    linear_->rhs(Gt[is], f, Rhs[is]);
    if (nonlinear_ != nullptr) {
      nonlinear_->nlps (Gt[is], f, Rhs[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }

    if (pars_->eqfix) Gnew[is]->copyFrom(G[is]);
    
    Gnew[is]->add_scaled(1., G[is], adt*dt_, Rhs[is]);
  }
  solver_->fieldSolve(Gnew, f);
}

void RungeKutta4::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  pars_->update_tprim(*t);

  partial(G, G,    f, GRhs,  G_q1, 0.5, true);
  partial(G, G_q1, f, GStar, G_q2, 0.5, false);

  // Do a partial accumulation of final update to save memory
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs[is]->add_scaled(dt_/6., GRhs[is], dt_/3., GStar[is]);
  }

  partial(G, G_q2, f, GStar, G_q1, 1., false);

  // This update is just to improve readability
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs[is]->add_scaled(1., GRhs[is], dt_/3., GStar[is]);
    
    GStar[is]->set_zero();
    linear_->rhs(G_q1[is], f, GStar[is]);
    if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1[is], f, GStar[is]);     
    
    G[is]->add_scaled(1., G[is], 1., GRhs[is], dt_/6., GStar[is]);

    /*
    partial(G, G_q2, f, G_q1, GStar, 1., false);

    linear_->rhs(GStar, f, G_q2);               
    if(nonlinear_ != nullptr) nonlinear_->nlps(GStar, f, G_q2);     
    
    G->add_scaled(1., G, 1., GRhs, dt_/3., G_q1, dt_/6., G_q2);
    */
    
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);
  *t += dt_;
}

