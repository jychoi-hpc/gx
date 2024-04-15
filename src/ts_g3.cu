#include "timestepper.h"
// #include "get_error.h"

G3::G3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), exb_(exb), dt_max(dt_in), dt_(dt_in),
  G_u1(nullptr), G_u2(nullptr)
{
  // new objects for temporaries
  G_u1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_u2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G_u1[is] = new MomentsG (pars_, grids_, is_glob);
    G_u2[is] = new MomentsG (pars_, grids_, is_glob);
  }
}

G3::~G3()
{
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G_u1[is]) delete G_u1[is];
    if (G_u2[is]) delete G_u2[is];
  }
  free(G_u1);
  free(G_u2);
}

void G3::EulerStep(MomentsG** G_u, MomentsG** GRhs, Fields* f, bool setdt)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    linear_->rhs(G_u[is], f, GRhs[is], dt_);

    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G_u[is], f, GRhs[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }

    G_u[is]->add_scaled(1., G_u[is], dt_, GRhs[is]);
  }
  solver_->fieldSolve(G_u, f);    
}

void G3::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->update_tprim(*t);
    G_u1[is]->update_tprim(*t);
  }

  // update flow shear terms if using ExB
  if (pars_->ExBshear) {
    exb_->flow_shear_shift(f, dt_);
    for(int is=0; is<grids_->Nspecies; is++) {
      exb_->flow_shear_g_shift(G[is]);
      exb_->flow_shear_g_shift(G_u1[is]);
    }
  }
  // end updates

  for(int is=0; is<grids_->Nspecies; is++) {
    G_u1[is]->copyFrom(G[is]);
    G_u2[is]->copyFrom(G[is]);
  }

  float onethird = 1./3.;
  float twothirds = 2./3.;  
  bool setdt = true;
  
  EulerStep(G_u1, G, f, setdt);
  setdt = false;

  EulerStep(G_u1, G, f, setdt);

  for(int is=0; is<grids_->Nspecies; is++) {
    G_u1[is]->add_scaled(0.75, G_u2[is], 0.25, G_u1[is]);
  }
  solver_->fieldSolve(G_u1, f);
  
  EulerStep(G_u1, G, f, setdt);
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->add_scaled(onethird, G_u2[is], twothirds, G_u1[is]);

    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  
}
