#include "timestepper.h"
#include <iostream>
// #include "get_error.h"

// ============= RK4 =============
SundialsStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  ctx()
{
	std::cout << "Using SUNDIALS for tiemstepping. This is fantastically unsupported and is probably wrong in all sorts of ways" << std::endl;
	// Set up over-arching SUNDIALS objects here
}

SundialsStepper::~SundialsStepper()
{
	// Free SUNDIALS Cruft
}

// ======== rk4  ==============

// We rely on the fact that G is the same one we were told about at the beginning.
void SundialsStepper::advance(double *t, MomentsG** G_, Fields* f)
{
	// Wrap MomentsG** in a GXVector
	GXVector G( G_, ctx );

	// update the gradients if they are evolving
	G.update_tprim( *t ); 
	G_q1.update_tprim( *t );
	G_q2.update_tprim( *t );
	// end updates

	partial(G, G,    f, GRhs,  G_q1, 0.5, true);
	partial(G, G_q1, f, GStar, G_q2, 0.5, false);

	// Do a partial accumulation of final update to save memory
	GRhs.LinearSum(dt_/6., GRhs, dt_/3., GStar);

	partial(G, G_q2, f, GStar, G_q1, 1., false);

	// This update is just to improve readability
	// start sync first, so that we can overlap it with computation below
	G_q1.sync();

	GRhs.LinearSum(1., GRhs, dt_/3., GStar);

	GStar.SetZero();

	if(nonlinear_ != nullptr) {
		for( int is = 0; is < grids_->Nspecies; ++is ) {
			nonlinear_->nlps(G_q1[is], f, GStar[is]);     
		}
	}

	G += GRhs;
	G.LinearSum(1., G, dt_/6., GStar);

	GStar.SetZero();

	for( int is = 0; is < grids_->Nspecies; ++is ) {
		linear_->rhs(G_q1[is], f, GStar[is], dt_);
	}

	G.LinearSum(1., G, dt_/6., GStar);

	if (forcing_ != nullptr) {
		for( int is = 0; is < grids_->Nspecies; ++is ) {
			forcing_->stir(G[is]);
		}
	}

	solver_->fieldSolve(G, f);
	*t += dt_;
}

