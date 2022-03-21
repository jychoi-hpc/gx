#include "timestepper.h"
// #include "get_error.h"

// This method is described in chapter 6 of David Ketcheson's thesis:
// https://www.davidketcheson.info/assets/papers/ketchesonphdthesis.pdf

// The option to dealias in kz presently requires extra FFTs and only works for
// systems without magnetic shear. It has been tested with slab ETG turbulence and
// works very well. It should not be assumed that this way of dealiasing in kz
// will work in any more general circumstances. 

// ============= K10,4 ============
Ketcheson10::Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), G_q1(nullptr), G_q2(nullptr)
{
  // new objects for temporaries
  G_q1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    G_q1[is] = new MomentsG (pars_, grids_, is);
    G_q2[is] = new MomentsG (pars_, grids_, is);
  }

  if (pars_->local_limit)                     { grad_par = new GradParallelLocal(grids_);
  } else if (pars_->boundary_option_periodic) { grad_par = new GradParallelPeriodic(grids_);
  } else {                                      grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
}

Ketcheson10::~Ketcheson10()
{
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G_q1[is]) delete G_q1[is];
    if (G_q2[is]) delete G_q2[is];
  }
  free(G_q1);
  free(G_q2);
  if (grad_par) delete grad_par;
}

void Ketcheson10::EulerStep(MomentsG** G_q1, MomentsG** GRhs, Fields* f, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs[is]->set_zero();
    linear_->rhs(G_q1[is], f, GRhs[is]);   if (pars_->dealias_kz) grad_par->dealias(GRhs[is]);
    
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G_q1[is], f, GRhs[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }
    if (pars_->dealias_kz) grad_par->dealias(GRhs[is]);

    G_q1[is]->add_scaled(1., G_q1[is], dt_/6., GRhs[is]);
  }
  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
}

void Ketcheson10::advance(double *t, MomentsG** G, Fields* f)
{
  bool setdt = true;

  // update the gradients if they are evolving
  pars_->update_tprim(*t);

  for(int is=0; is<grids_->Nspecies; is++) {
    G_q1[is] -> copyFrom(G[is]);
    G_q2[is] -> copyFrom(G[is]);
  }

  for(int i=1; i<6; i++) { EulerStep(G_q1, G, f, setdt);    setdt = false;}

  for(int is=0; is<grids_->Nspecies; is++) {
    G_q2[is] -> add_scaled(0.04, G_q2[is], 0.36, G_q1[is]);
    G_q1[is] -> add_scaled(15, G_q2[is], -5, G_q1[is]);
  }

  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  
  for(int i=6; i<10; i++) EulerStep(G_q1, G, f, setdt);
  
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->set_zero();
    linear_->rhs(G_q1[is], f, G[is]);
    if (pars_->dealias_kz) grad_par->dealias(G[is]);
    
    if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1[is], f, G[is]);
    if (pars_->dealias_kz) grad_par->dealias(G[is]);
    
    G[is]->add_scaled(1., G_q2[is], 0.6, G_q1[is], 0.1*dt_, G[is]);
    
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  *t += dt_;
}

