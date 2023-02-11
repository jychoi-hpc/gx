#include "timestepper.h"
// #include "get_error.h"

K2::K2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), 
  G_q1(nullptr), G_q2(nullptr)
{
  stages_ = pars_->stages;
  sm1inv = (double) 1./((double) stages_-1);
  sinv   = (double) 1./((double) stages_);
  
  // new objects for temporaries
  G_q1 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  G_q2 = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G_q1[is] = new MomentsG (pars_, grids_, is_glob);
    G_q2[is] = new MomentsG (pars_, grids_, is_glob);
  }
}

K2::~K2()
{
  for(int is=0; is<grids_->Nspecies; is++) {
    if (G_q1[is]) delete G_q1[is];
    if (G_q2[is]) delete G_q2[is];
  }
  free(G_q1);
  free(G_q2);
}

void K2::EulerStep(MomentsG** G_q1, MomentsG** GRhs, Fields* f, bool setdt)
{  
  for(int is=0; is<grids_->Nspecies; is++) {
    linear_->rhs(G_q1[is], f, GRhs[is]);

    if(nonlinear_ != nullptr) {
      nonlinear_->nlps(G_q1[is], f, GRhs[is]);
      if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
    }

    G_q1[is]->add_scaled(1., G_q1[is], dt_*sm1inv, GRhs[is]);
  }
  solver_->fieldSolve(G_q1, f);    
}

void K2::FinalStep(MomentsG** G_q1, MomentsG** G_q2, MomentsG** GRhs, Fields* f)
{  
  for(int is=0; is<grids_->Nspecies; is++) {
    linear_->rhs(G_q1[is], f, GRhs[is]);
    if(nonlinear_ != nullptr) nonlinear_->nlps(G_q1[is], f, GRhs[is]);

    double sm1 = (double) stages_ - 1;  
    GRhs[is]->add_scaled(sm1*sinv, G_q1[is], sinv, G_q2[is], dt_*sinv, GRhs[is]);
  }
  // no field solve required here
}

void K2::advance(double *t, MomentsG** G, Fields* f)
{
  // update the gradients if they are evolving
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]   -> update_tprim(*t);
    G_q1[is]-> update_tprim(*t);
    G_q2[is]-> update_tprim(*t);
  }
  // end updates

  for(int is=0; is<grids_->Nspecies; is++) {
    G_q1[is] -> copyFrom(G[is]);
    G_q2[is] -> copyFrom(G[is]);
  }

  bool setdt = true;
  
  for(int i=1; i<stages_; i++) {
    EulerStep(G_q1, G, f, setdt);
    setdt = false;
  }

  FinalStep(G_q1, G_q2, G, f); // returns G 
  for(int is=0; is<grids_->Nspecies; is++) {
    if (forcing_ != nullptr) forcing_->stir(G[is]);
  }
  
  solver_->fieldSolve(G, f);
  *t += dt_;
}

