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
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is] = new MomentsG (pars, grids, is);
  }
}

RungeKutta2::~RungeKutta2()
{
  if (GRhs)  delete GRhs;
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G1[is])    delete G1[is];
  }
  free(G1);
}

// ======== rk2  ==============
void RungeKutta2::EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** G, MomentsG* GRhs,
			    Fields* f, double adt, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs->set_zero();
    linear_->rhs(G0[is], f, GRhs); 
  
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G0[is], f, GRhs);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }

    if (pars_->eqfix) G1[is]->copyFrom(G[is]);   

    G1[is]->add_scaled(1., G[is], adt*dt_, GRhs); 
  }
}

void RungeKutta2::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  pars_->update_tprim(*t);

  EulerStep (G1, G, G, GRhs, f, 0.5, true);    solver_->fieldSolve(G1, f);
  EulerStep (G, G1, G, GRhs, f, 1.0, false);   

  if (forcing_ != nullptr) {
    for(int is=0; is<grids_->Nspecies; is++) {
      forcing_->stir(G[is]);  
    }
  }
   
  solver_->fieldSolve(G, f);

  *t += dt_;
}

