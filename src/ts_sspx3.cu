#include "timestepper.h"
// #include "get_error.h"

/*
This method was set up by Greg Hammett (and a student, see below), 
with a minor correction from Dorland.

The derivation of the method is laid out in the file SSPX3-4_2016.12.01.pdf 
in the docs/files directory of this repo, with a correction needed for the 
value of w3 on page 5. This required correction appears in the specification 
of w3 for this method, which appears in ../include/timestepper.h. 

The notes in the PDF above are from math student Federico Pasqualotto. 
*/

// ======= SSPx3 =======
SSPx3::SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), exb_(exb), dt_max(pars->dt_max), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr), G3(nullptr)
{
  
  // new objects for temporaries
  GRhs = new MomentsG (pars_, grids_);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G3 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G1[is] = new MomentsG (pars_, grids_, is_glob);
    G2[is] = new MomentsG (pars_, grids_, is_glob);
    G3[is] = new MomentsG (pars_, grids_, is_glob);
  }

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
  else if (pars_->boundary_option_periodic) {
    grad_par = new GradParallelPeriodic(grids_);
  }
  else if (pars_->nonTwist) {
    grad_par = new GradParallelNTFT(pars_, grids_);
  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }
  
}

SSPx3::~SSPx3()
{
  if (GRhs) delete GRhs;
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G1[is]) delete G1[is];
    if (G2[is]) delete G2[is];
    if (G3[is]) delete G3[is];
  }
  free(G1);
  free(G2);
  free(G3);
  if (grad_par) delete grad_par;
}

// ======== SSPx3  ==============
void SSPx3::EulerStep(MomentsG** G1, MomentsG** G, MomentsG* GRhs, Fields* f, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    G[is]->sync();

    if (pars_->eqfix) G1[is]->copyFrom(G[is]);   

    // compute timestep (if necessary)
    if (setdt && is==0 && !pars_->fixed_dt) { // dt will be computed same for all species, so just do first time through species loop
      linear_->get_max_frequency(omega_max);
      if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
      double wmax = 0.;
      for(int i=0; i<3; i++) wmax += omega_max[i];
	  double dt_guess = cfl_fac*pars_->cfl/wmax;
      dt_ = min( max(dt_guess,pars_->dt_min), dt_max);
    }

    // compute and increment nonlinear term
    GRhs->set_zero();
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G[is], f, GRhs);
    }
    G1[is]->add_scaled(1., G[is], adt*dt_, GRhs);

    // compute and increment linear term
    GRhs->set_zero();
    linear_->rhs(G[is], f, GRhs, dt_);  if (pars_->dealias_kz) grad_par->dealias(GRhs);

    G1[is]->add_scaled(1., G1[is], adt*dt_, GRhs);
  }
}

void SSPx3::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is] -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
    G2[is]-> update_tprim(*t);
  }

  // update flow shear terms if using ExB
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_);
    for(int is=0; is<grids_->Nspecies; is++) {
      exb_->flow_shear_g_shift(G[is]);
      exb_->flow_shear_g_shift(G1[is]);
      exb_->flow_shear_g_shift(G2[is]);
    }
  }

  // end of updates
  
  EulerStep (G1, G , GRhs, f, true);  
  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);
  EulerStep (G2, G1, GRhs, f, false); 

  for(int is=0; is<grids_->Nspecies; is++) {
    G2[is]->add_scaled((1.-w1), G[is], (w1-1.), G1[is], 1., G2[is]);
  }
  solver_->fieldSolve(G2, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  EulerStep (G3, G2, GRhs, f, false);

  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled((1.-w2-w3), G[is], w3, G1[is], (w2-1.), G2[is], 1., G3[is]);
  
    if (forcing_ != nullptr) forcing_->stir(G[is]);  
    G[is]->mask();
  }
  solver_->fieldSolve(G, f);          if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}

