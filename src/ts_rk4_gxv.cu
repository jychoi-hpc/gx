#include "timestepper.h"
// #include "get_error.h"

// ============= RK4 =============
GXVRK4::GXVRK4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in),
  ctx(), GStar(pars, grids, ctx), GRhs(pars, grids, ctx), G_q1(pars, grids, ctx), G_q2(pars, grids, ctx)
{

}

GXVRK4::~GXVRK4()
{
}

// ======== rk4  ==============

void GXVRK4::partial(GXVector & G, GXVector & Gt, Fields *f, GXVector & Rhs, GXVector & Gnew, double adt, bool setdt)
{
	// start sync first, so that we can overlap it with computation below
	Gt.sync();

	if (pars_->eqfix) 
		Gnew = G;

	// compute timestep (if necessary)
	if (setdt) {
		linear_->get_max_frequency(omega_max);
		if (nonlinear_ != nullptr) nonlinear_->get_max_frequency(f, omega_max);
		double wmax = 0.;
		for(int i=0; i<3; i++) wmax += omega_max[i];
		dt_ = min(cfl_fac*pars_->cfl/wmax, dt_max);
	}

	// compute and increment nonlinear term
	Rhs.SetZero();

	if (nonlinear_ != nullptr) {
		for( int is = 0; is < grids_->Nspecies; ++is ) {
			nonlinear_->nlps (Gt[is], f, Rhs[is]);
		}
	}

	Gnew[is]->add_scaled(1., G[is], adt*dt_, Rhs[is]);

	// compute and increment linear term
	Rhs.SetZero();
	for( int i = 0; i < grids_->Nspecies; ++i)
		linear_->rhs(Gt[is], f, Rhs[is], dt_);
	Gnew.LinearSum(1., Gnew, adt*dt_, Rhs);

	// need to recompute and save Rhs for intermediate steps
	Rhs.LinearSum(1./(adt*dt_), Gnew, -1./(adt*dt_), G);

	// compute new fields
	solver_->fieldSolve(Gnew, f);
}

void GXVRK4::advance(double *t, MomentsG** G_, Fields* f)
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

	G.LinearSum(1., G, 1., GRhs, dt_/6., GStar);

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

