#include "timestepper.h"
#include <iostream>

#include <stdexcept>

#include <arkode/arkode_erkstep.h>

// #include "get_error.h"

// ============= RK4 =============
SundialsStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), exb_(exb), dt_(dt_in), ctx(), ERKStepMem(nullptr), gInternal(nullptr), fields_(nullptr)
{
	std::cout << "Using SUNDIALS for timestepping. This is fantastically unsupported and is probably wrong in all sorts of ways" << std::endl;
}

SundialsStepper::~SundialsStepper()
{
	if( ERKStepMem != nullptr )
		ERKStepFree( &ERKStepMem );
	if( gInternal != nullptr )
		delete gInternal;
	if( gTmp != nullptr )
		delete gTmp;
}


void SundialsStepper::Initialise( MomentsG** g0, double t0 )
{
	if( ERKStepMem != nullptr ) {
		throw std::runtime_error("Double Initialisation of Sundials Timestepper. Aborting.");
	}
	
	// This creates a GXVector that is a view of the data in the MomentsG** but 
	// does not *own* the data. Thus deleting this pointer will not free the underlying MomentsG
	gInternal = new GXVector( g0, ctx );
	gTmp = new GXVector( g0, ctx );

	// Wrap GXVector in an NVector

	gInternalNV = gInternal->asNVector();

	std::cout << "Initialising ERKStep" << std::endl;
	ERKStepMem = ERKStepCreate( SundialsStepper::SundialsF, t0, gInternalNV, ctx );
	if( ERKStepMem == nullptr )
		throw std::runtime_error("Unable to allocate SUNDIALS Memory. ABORT.");

	int retval;

	retval = ERKStepSStolerances( ERKStepMem, reltol, abstol );
	if( retval != ARK_SUCCESS ) {
		throw std::runtime_error("Internal SUNDIALS Error in ERKStepSStolerances.");
	}
	
	// Use the 4th order Zonnefeld method
	retval = ERKStepSetTableNum( ERKStepMem, ARKODE_ZONNEVELD_5_3_4 );

	if( retval != ARK_SUCCESS ) {
		throw std::runtime_error("Internal SUNDIALS Error in ERKStepSetTableNum.");
	}

	ERKStepSetUserData( ERKStepMem, static_cast<void*>(this) );
	ERKStepSetInitStep( ERKStepMem, dt_ );
	ARKStepSetFixedStep( ERKStepMem, dt_ );
}



// We rely on the fact that G is the same one we were told about initially
void SundialsStepper::advance(double *t, MomentsG** G_, Fields* f)
{
	if( ERKStepMem == nullptr ) {
		// First timestep. Do initialisation.
		Initialise( G_, *t );
	}

	double t_out = *t + dt_; // Try to advance one `time step', possibly using multiple internal steps

	int retval;

	fields_ = f;

	retval = ERKStepEvolve( ERKStepMem, t_out, gInternalNV, t, ARK_NORMAL );

	if( retval != ARK_SUCCESS ) {
		throw std::runtime_error("Error in ERKStepEvolve.");
	}

	// Set fields_ to be the fields consistent with the final state (for diagnostics etc)
	solver_->fieldSolve( *gInternal, fields_ );
}

int SundialsStepper::SundialsF( sunrealtype t, N_Vector y, N_Vector ydot, void* userdata )
{
	GXVector *g = reinterpret_cast<GXVector*>( y->content );
	GXVector *gdot = reinterpret_cast<GXVector*>( ydot->content );
	return reinterpret_cast<SundialsStepper*>( userdata )->SundialsRHS( t, g, gdot );
}

int SundialsStepper::SundialsRHS( double time, GXVector *g, GXVector * gdot )
{
	// Make sure fields are evaluated at this current g
	solver_->fieldSolve( *g, fields_ );

	// compute nonlinear term and write to gdot
	gdot->SetZero();

	if (nonlinear_ != nullptr) {
		for( int is = 0; is < grids_->Nspecies; ++is ) {
			nonlinear_->nlps ( (*g)[is], fields_, (*gdot)[is]);
		}
	}

	// Accumulate Linear Terms into gTmp
	gTmp->SetZero();

	// compute and accumulate linear term
	for( int is = 0; is < grids_->Nspecies; ++is)
		linear_->rhs( (*g)[is], fields_, (*gTmp)[is], dt_ );

	*gdot += *gTmp; // Add NL + L

	return 0;

}

