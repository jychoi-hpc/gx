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
			 Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), exb_(exb), dt_max(pars->dt_max), dt_(dt_in), G_q1(nullptr), G_q2(nullptr), Gtmp(nullptr)
{
  // new objects for temporaries
  Gtmp = new MomentsG (pars_, grids_);
  G_q1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G_q1[is] = new MomentsG (pars_, grids_, is_glob);
    G_q2[is] = new MomentsG (pars_, grids_, is_glob);
  }

  if(pars_->dealias_kz) {
    if (pars_->local_limit)                     { grad_par = new GradParallelLocal(grids_);
    } else if (pars_->boundary_option_periodic) { grad_par = new GradParallelPeriodic(grids_);
    } else if (pars_->nonTwist)                 { grad_par = new GradParallelNTFT(pars_, grids_);
    } else {                                      grad_par = new GradParallelLinked(pars_, grids_);
    }
  }
}

Ketcheson10::~Ketcheson10()
{
  if (Gtmp) delete Gtmp;
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G_q1[is]) delete G_q1[is];
    if (G_q2[is]) delete G_q2[is];
  }
  free(G_q1);
  free(G_q2);
  if (grad_par) delete grad_par;
}

void Ketcheson10::EulerStep(MomentsG** G_q1, MomentsG** GRhs, MomentsG* Gtmp, Fields* f, bool setdt)
{
  if (setdt) dt_ = dt_max;
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    G_q1[is]->sync();

    // compute timestep (if necessary)
    if (setdt && is==0) { // dt will be computed same for all species, so just do first time through species loop
      linear_->get_max_frequency(omega_max);
      if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
      double wmax = 0.;
      for(int i=0; i<3; i++) wmax += omega_max[i];
      dt_ = min(pars_->cfl/wmax, dt_max);
    }

    // compute and increment nonlinear term
    GRhs[is]->set_zero();
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G_q1[is], f, GRhs[is]);
    }
    Gtmp->add_scaled(1., G_q1[is], dt_/6., GRhs[is]);

    // compute and increment linear term
    GRhs[is]->set_zero();
    linear_->rhs(G_q1[is], f, GRhs[is], dt_);  if (pars_->dealias_kz) grad_par->dealias(GRhs[is]);

    G_q1[is]->add_scaled(1., Gtmp, dt_/6., GRhs[is]);
  }

  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
}

void Ketcheson10::advance(double *t, MomentsG** G, Fields* f)
{
  bool setdt = true;

  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G_q1[is]-> update_tprim(*t);
  }
  
  // update flow shear terms if using ExB
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_);
    for(int is=0; is<grids_->Nspecies; is++) {
      //exb_->flow_shear_g_shift(G[is]); // only G_q1? //JMH
      exb_->flow_shear_g_shift(G_q1[is]);
    }
  }
  // end updates

  for(int is=0; is<grids_->Nspecies; is++) {
    G_q1[is] -> copyFrom(G[is]);
    G_q2[is] -> copyFrom(G[is]);
  }

  for(int i=1; i<6; i++) { EulerStep(G_q1, G, Gtmp, f, setdt);    setdt = false;}

  for(int is=0; is<grids_->Nspecies; is++) {
    G_q2[is] -> add_scaled(0.04, G_q2[is], 0.36, G_q1[is]);
    G_q1[is] -> add_scaled(15, G_q2[is], -5, G_q1[is]);
  }

  solver_->fieldSolve(G_q1, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  
  for(int i=6; i<10; i++) EulerStep(G_q1, G, Gtmp, f, setdt);
  
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    G_q1[is]->sync();

    // compute and increment nonlinear term
    G[is]->set_zero();
    if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1[is], f, G[is]);
    if (pars_->dealias_kz) grad_par->dealias(G[is]);
    Gtmp->add_scaled(1., G_q2[is], 0.6, G_q1[is], 0.1*dt_, G[is]);

    // compute and increment linear term
    G[is]->set_zero();
    linear_->rhs(G_q1[is], f, G[is], dt_);
    if (pars_->dealias_kz) grad_par->dealias(G[is]);
    G[is]->add_scaled(1., Gtmp, 0.1*dt_, G[is]);
    
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);  if (pars_->dealias_kz) grad_par->dealias(f->phi);
  *t += dt_;
}

