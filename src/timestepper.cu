#include "timestepper.h"
#include "get_error.h"

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in) :
  linear_(linear), solver_(solver), grids_(grids), forcing_(forcing)
{
  // new objects for temporaries
  mStar = new Moments(grids);
  mRhs = new Moments(grids);
  dt_ = dt_in;
}

RungeKutta2::~RungeKutta2()
{
  delete mStar;
  delete mRhs;
}

int RungeKutta2::advance(double *t, Moments* m, Fields* f) {
  linear_->rhs(m, f, mRhs);
  mStar->add_scaled(1., m, dt_/2., mRhs);
  solver_->fieldSolve(mStar, f);

  linear_->rhs(mStar, f, mRhs);
  mStar->add_scaled(1., m, dt_, mRhs);

  if (forcing_ != NULL) {
    forcing_->stir(mStar);
  }

  m->copyFrom(mStar);

  solver_->fieldSolve(m, f);

  linear_->dissipation(m, f, dt_);

  solver_->fieldSolve(m, f);

  *t+=dt_;
  return 0;
}

// ============= RK4 =============
RungeKutta4::RungeKutta4(Linear *linear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in) :
  linear_(linear), solver_(solver), grids_(grids), forcing_(forcing)
{
  // new objects for temporaries
  mStar = new Moments(grids);
  mRhs1 = new Moments(grids);
  mRhs2 = new Moments(grids);
  mRhs3 = new Moments(grids);
  mRhs4 = new Moments(grids);
  dt_ = dt_in;
}

RungeKutta4::~RungeKutta4()
{
  delete mStar;
  delete mRhs1;
  delete mRhs2;
  delete mRhs3;
  delete mRhs4;
}

int RungeKutta4::advance(double *t, Moments* m, Fields* f) {
  linear_->rhs(m, f, mRhs1);
  mStar->add_scaled(1., m, dt_/2., mRhs1);
  solver_->fieldSolve(mStar, f);

  linear_->rhs(mStar, f, mRhs2);
  mStar->add_scaled(1., m, dt_/2., mRhs2);
  solver_->fieldSolve(mStar, f);

  linear_->rhs(mStar, f, mRhs3);
  mStar->add_scaled(1., m, dt_/2., mRhs3);
  solver_->fieldSolve(mStar, f);

  linear_->rhs(mStar, f, mRhs4);
  mStar->add_scaled(1., m, dt_/6., mRhs1, dt_/3., mRhs2, dt_/3., mRhs3, dt_/6., mRhs4);
 
  if (forcing_ != NULL) {
    forcing_->stir(mStar);
  }
  
  m->copyFrom(mStar);
  solver_->fieldSolve(m, f);
  *t+=dt_;
  return 0;
}
