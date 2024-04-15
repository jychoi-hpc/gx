#include "timestepper.h"
// #include "get_error.h"

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), exb_(exb), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr)
{
  // new objects for temporaries
  GRhs  = new MomentsG (pars, grids);
  G1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G1[is] = new MomentsG (pars, grids, is_glob);
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
    // start sync first, so that we can overlap it with computation below
    G0[is]->sync();

    if (pars_->eqfix) G1[is]->copyFrom(G[is]);   

    // compute timestep (if necessary)
    if (setdt && is==0) { // dt will be computed same for all species, so just do first time through species loop
      linear_->get_max_frequency(omega_max);
      if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
      double wmax = 0.;
      for(int i=0; i<3; i++) wmax += omega_max[i];
      dt_ = min(cfl_fac*pars_->cfl/wmax, dt_max);
    }

    GRhs->set_zero();
    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G0[is], f, GRhs);
    }
    G1[is]->add_scaled(1., G[is], adt*dt_, GRhs); 

    GRhs->set_zero();
    linear_->rhs(G0[is], f, GRhs, dt_); 
  
    G1[is]->add_scaled(1., G1[is], adt*dt_, GRhs); 
  }
}

void RungeKutta2::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]   -> update_tprim(*t);
    G1[is]-> update_tprim(*t);
  }
  
  // update flow shear terms if using ExB
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_);
    for(int is=0; is<grids_->Nspecies; is++) {
      exb_->flow_shear_g_shift(G[is]);
      exb_->flow_shear_g_shift(G1[is]);
    }
  }
  // end updates

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

