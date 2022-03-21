#include "timestepper.h"

// ======= SSPx2 =======
SSPx2::SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr)
{
  // new objects for temporaries
  GRhs  = new MomentsG (pars, grids);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is] = new MomentsG (pars_, grids_, is);
    G2[is] = new MomentsG (pars_, grids_, is);
  }
}

SSPx2::~SSPx2()
{
  if (GRhs)  delete GRhs;
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G1[is]) delete G1[is];
    if (G2[is]) delete G2[is];
  }
  free(G1);
  free(G2);
}

// ======== SSPx2  ==============
void SSPx2::EulerStep(MomentsG** G1, MomentsG** G, MomentsG* GRhs, Fields* f, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs->set_zero();
    linear_->rhs(G[is], f, GRhs);

    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, GRhs);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }

    if (pars_->eqfix) G1[is]->copyFrom(G[is]);   
    G1[is]->add_scaled(1., G[is], dt_/sqrt(2.), GRhs);
  }
}

void SSPx2::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  pars_->update_tprim(*t);
  
  EulerStep (G1, G, GRhs, f, true); 
  solver_->fieldSolve(G1, f);

  EulerStep (G2, G1, GRhs, f, false);

  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(2.-sqrt(2.), G[is], sqrt(2.)-2., G1[is], 1., G2[is]);
    
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
  }

  solver_->fieldSolve(G, f);

  *t += dt_;
}
