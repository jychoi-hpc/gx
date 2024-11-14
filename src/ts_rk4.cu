#include "timestepper.h"
// #include "get_error.h"

// ============= RK4 =============
RungeKutta4::RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), exb_(exb), dt_max(pars->dt_max), dt_(dt_in),
  GStar(nullptr), GRhs(nullptr), G_q1(nullptr), G_q2(nullptr)
{
  GStar = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  GRhs = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    GStar[is] = new MomentsG (pars_, grids_, is_glob);
    GRhs[is] = new MomentsG (pars_, grids_, is_glob);
    G_q1[is] = new MomentsG (pars_, grids_, is_glob);
    G_q2[is] = new MomentsG (pars_, grids_, is_glob);
  }
  if( pars->fixed_dt )
	set_dt = false;
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

void RungeKutta4::set_timestep( Fields * f )
{
	linear_->get_max_frequency(omega_max);
	if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
	double wmax = 0.;
	for(int i=0; i<3; i++) wmax += omega_max[i];
	double dt_guess = cfl_fac*pars_->cfl/wmax;
    dt_ = min( max(dt_guess,pars_->dt_min), dt_max);
}

void RungeKutta4::partial(MomentsG** G, MomentsG** Gt, Fields *f, MomentsG** Rhs, MomentsG **Gnew, double adt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    Gt[is]->sync();

    if (pars_->eqfix) Gnew[is]->copyFrom(G[is]);

    // compute and increment nonlinear term
    Rhs[is]->set_zero();
    if (nonlinear_ != nullptr) {
      nonlinear_->nlps (Gt[is], f, Rhs[is]);
    }
	 // Gnew = G + adt*(dt_)*Rhs
    Gnew[is]->add_scaled(1., G[is], adt*dt_, Rhs[is]);

    // compute and increment linear term
    Rhs[is]->set_zero();
    linear_->rhs(Gt[is], f, Rhs[is], dt_);
	 // Gnew += adt*(dt_)*Rhs
    Gnew[is]->add_scaled(1., Gnew[is], adt*dt_, Rhs[is]);
  
    // need to recompute and save Rhs for intermediate steps
	 // Rhs = (Gnew - G)/(adt*(dt_))
    Rhs[is]->add_scaled(1./(adt*dt_), Gnew[is], -1./(adt*dt_), G[is]);
  }

  // compute new fields
  solver_->fieldSolve(Gnew, f);
}

void RungeKutta4::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]   -> update_tprim(*t);
    G_q1[is]-> update_tprim(*t);
    G_q2[is]-> update_tprim(*t);
  }
  
  // if we're changing the timestep, set it now
  if( set_dt )
	  set_timestep( f );

  // dt_ now contains the new timestep
  
  // This constitutes evaluating the RHS at t & g=G, storing into GRhs, and putting G + GRhs in G_q1
  partial(G, G,    f, GRhs,  G_q1, 0.5);
  
  // update flow shear terms to t = t + dt_/2 if using ExB
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_ * 0.5);
    for(int is=0; is<grids_->Nspecies; is++) {
      exb_->flow_shear_g_shift(G[is]);
      exb_->flow_shear_g_shift(GRhs[is]);
      exb_->flow_shear_g_shift(G_q1[is]);
    }
  }
  // end updates

  // This evaluates RHS at t + dt/2 & g=G_q1, into GStar and putting G+GStar in G_q2
  partial(G, G_q1, f, GStar, G_q2, 0.5);

  // Do a partial accumulation of final update to save memory
  for(int is=0; is<grids_->Nspecies; is++) {
    GRhs[is]->add_scaled(dt_/6., GRhs[is], dt_/3., GStar[is]);
  }

  // Second evaluation at t + dt/2, now at g = G_q2, storing RHS in GStar, and G + GStar back in G_q1
  partial(G, G_q2, f, GStar, G_q1, 1.);

  // Shift forwards to t = t + dt_
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_ * 0.5);
    for(int is=0; is<grids_->Nspecies; is++) {
      exb_->flow_shear_g_shift(G[is]);
      exb_->flow_shear_g_shift(GRhs[is]);
      exb_->flow_shear_g_shift(GStar[is]);
      exb_->flow_shear_g_shift(G_q1[is]);
    }
  }

  // This update is just to improve readability
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    G_q1[is]->sync();

    GRhs[is]->add_scaled(1., GRhs[is], dt_/3., GStar[is]);
    
    GStar[is]->set_zero();
    if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1[is], f, GStar[is]);     
    G[is]->add_scaled(1., G[is], 1., GRhs[is], dt_/6., GStar[is]);

    GStar[is]->set_zero();
    linear_->rhs(G_q1[is], f, GStar[is], dt_);
    
    G[is]->add_scaled(1., G[is], dt_/6., GStar[is]);
    
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);
  *t += dt_;
}

