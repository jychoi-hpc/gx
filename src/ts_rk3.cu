#include "timestepper.h"
// #include "get_error.h"

// ============= RK3 =============
// Heun's method
RungeKutta3::RungeKutta3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  GRhs1(nullptr), GRhs2(nullptr), G_q1(nullptr), G_q2(nullptr)
{
  GRhs1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  GRhs2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q1  = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2  = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    GRhs1[is] = new MomentsG (pars_, grids_, is_glob);
    GRhs2[is] = new MomentsG (pars_, grids_, is_glob);
    G_q1[is]  = new MomentsG (pars_, grids_, is_glob);
    G_q2[is]  = new MomentsG (pars_, grids_, is_glob);
  }
}

RungeKutta3::~RungeKutta3()
{
  for(int is=0; is<grids_->Nspecies; is++) {
    if (GRhs1[is]) delete GRhs1[is];
    if (GRhs2[is]) delete GRhs2[is];
    if (G_q1[is]) delete G_q1[is];
    if (G_q2[is]) delete G_q2[is];
  }
  free(GRhs1);
  free(GRhs2);
  free(G_q1);
  free(G_q2);
}

// ======== rk3  ==============

// partial(G, Gt, f, Rhs, Gnew):
// Rhs = 0
// Rhs = Nonlin(Gt)
// Gnew = G + a*dt*Rhs 
// Rhs = 0
// Rhs = Lin(Gt)
// Gnew = Gnew + a*dt*Rhs
// can have G == Gt, otherwise all other arguments must be unique
void RungeKutta3::partial(MomentsG** G, MomentsG** Gt, Fields *f, MomentsG** Rhs, MomentsG **Gnew, double adt, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    // start sync first, so that we can overlap it with computation below
    Gt[is]->sync();

    if (pars_->eqfix) Gnew[is]->copyFrom(G[is]);

    // compute timestep (if necessary)
    if (setdt && is==0) { // dt will be computed same for all species, so just do first time through species loop
      linear_->get_max_frequency(omega_max);
      if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
      double wmax = 0.;
      for(int i=0; i<3; i++) wmax += omega_max[i];
      // Print the two quantities before calculating the minimum
      //std::cout << "cfl_fac*pars_->cfl/wmax: " << cfl_fac * pars_->cfl / wmax << std::endl;
      //std::cout << "dt_max: " << dt_max << std::endl;
      //std::cout << "cfl_fac: " << cfl_fac << std::endl;
      dt_ = min(cfl_fac*pars_->cfl/wmax, dt_max);
    }

    Rhs[is]->set_zero();
    if (nonlinear_ != nullptr) {
      nonlinear_->nlps (Gt[is], f, Rhs[is]);
    }
    Gnew[is]->add_scaled(1., G[is], adt*dt_, Rhs[is]);

    // compute and increment linear term
    Rhs[is]->set_zero();
    linear_->rhs(Gt[is], f, Rhs[is], dt_);
    Gnew[is]->add_scaled(1., Gnew[is], adt*dt_, Rhs[is]);
  
    // need to recompute and save Rhs for intermediate steps
    Rhs[is]->add_scaled(1./(adt*dt_), Gnew[is], -1./(adt*dt_), G[is]);
  }
}

void RungeKutta3::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]   -> update_tprim(*t);
    G_q1[is]-> update_tprim(*t);
    G_q2[is]-> update_tprim(*t);
  }
  // end updates

  // GRhs1 = RHS(G)
  // G_q1 = G + dt/3*GRhs1
  partial(G, G, f, GRhs1, G_q1, 1./3., true); 
  solver_->fieldSolve(G_q1, f);

  // GRhs2 = RHS(G_q1)
  // G_q2 = G + 2*dt/3*GRhs2
  partial(G, G_q1, f, GRhs2, G_q2, 2./3., false);
  solver_->fieldSolve(G_q2, f);

  // GRhs2 = RHS(G_q2)
  // G_q1 = G + 3*dt/4*GRhs2
  partial(G, G_q2, f, GRhs2, G_q1, 0.75, false);

  // G = G_q1 + dt/4*GRhs1
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(1., G_q1[is], 0.25*dt_, GRhs1[is]);
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }

  solver_->fieldSolve(G, f);
  *t += dt_;
}

