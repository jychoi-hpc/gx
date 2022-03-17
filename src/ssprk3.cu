
// ======= SSP-RK3 =======
SSPRK3::SSPRK3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in), GRhs(nullptr), G1(nullptr), G2(nullptr)
{
  
  // new objects for temporaries
  GRhs  = new MomentsG (pars_, grids_);
  G1    = new MomentsG (pars_, grids_);
  G2    = new MomentsG (pars_, grids_);

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
  else if (pars_->boundary_option_periodic) {
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
  
}

SSPRK3::~SSPRK3()
{
  if (GRhs)  delete GRhs;
  if (G1)    delete G1; 
  if (G2)    delete G2; 
  if (grad_par) delete grad_par;
}

void SSPRK3::EulerStep(MomentsG* G1, MomentsG* G, MomentsG* GRhs, Fields* f, bool setdt)
{
  linear_->rhs(G, f, GRhs);  if (pars_->dealias_kz) grad_par->dealias(GRhs);

  if(nonlinear_ != nullptr) {
    nonlinear_->nlps(G, f, GRhs);
    if (setdt) dt_ = nonlinear_->cfl(f, dt_max);
  }
  if (pars_->dealias_kz) grad_par->dealias(GRhs);

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, dt_, GRhs);
}

void SSPRK3::advance(double *t, MomentsG* G, Fields* f)
{
  // update the gradients if they are evolving
  G -> update_tprim(*t); 
  G1-> update_tprim(*t); 
  G2-> update_tprim(*t); 
  // end updates
  
  // stage 1
  EulerStep (G1, G , GRhs, f, true);  
  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 2
  EulerStep (G2, G1, GRhs, f, false); 
  G1->add_scaled(0.75, G, 0.25, G2);
  solver_->fieldSolve(G1, f);         if (pars_->dealias_kz) grad_par->dealias(f->phi);

  // stage 3
  EulerStep (G2, G1, GRhs, f, false);
  G->add_scaled(1.0/3.0, G, 2.0/3.0, G2);
  
  if (forcing_ != nullptr) forcing_->stir(G);  
  G->mask();
  solver_->fieldSolve(G, f);          if (pars_->dealias_kz) grad_par->dealias(f->phi);

  *t += dt_;
}

