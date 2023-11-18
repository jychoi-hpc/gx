#include "timestepper.h"

// ======= SSP-RK3 =======
SSPRK3::SSPRK3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr)
{
  
  // new objects for temporaries
  GRhs  = new MomentsG (pars_, grids_);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is] = new MomentsG (pars_, grids_, is);
    G2[is] = new MomentsG (pars_, grids_, is);
  }

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
  else if (pars_->boundary_option_periodic) {
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }
  
}

SSPRK3::~SSPRK3()
{
  if (GRhs)  delete GRhs;
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G1[is]) delete G1[is];
    if (G2[is]) delete G2[is];
  }
  free(G1);
  free(G2);
  if (grad_par) delete grad_par;
}

void SSPRK3::EulerStep(MomentsG** G1, MomentsG** G, MomentsG* GRhs, Fields* f, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs->set_zero();
    linear_->rhs(G[is], f, GRhs, dt_);  if (pars_->dealias_kz) grad_par->dealias(GRhs);

    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, GRhs);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
    if (pars_->dealias_kz) grad_par->dealias(GRhs);

    if (pars_->eqfix) G1[is]->copyFrom(G[is]);   
    G1[is]->add_scaled(1., G[is], dt_, GRhs);

  }

}

void SSPRK3::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
    G2[is]-> update_tprim(*t);
  }  
  // stage 1
  EulerStep (G1, G , GRhs, f, true);  
  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 2
  EulerStep (G2, G1, GRhs, f, false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G1[is]->add_scaled(0.75, G[is], 0.25, G2[is]);
  }

  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  EulerStep (G2, G1, GRhs, f, false);
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1.0/3.0, G[is], 2.0/3.0, G2[is]);
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();

  }
  
  solver_->fieldSolve(G, f);          if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}

