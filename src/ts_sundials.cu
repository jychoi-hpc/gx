#include "timestepper.h"
#include <iostream>

#include <stdexceot>

#include <arkode/arkode_erkstep.h>

// #include "get_error.h"

// ============= RK4 =============
SundialsStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_(dt_in), ctx(), ERKStepMem(nullptr), gInternal(nullptr), fields_(nullptr)
{
	std::cout << "Using SUNDIALS for tiemstepping. This is fantastically unsupported and is probably wrong in all sorts of ways" << std::endl;
}

SundialsStepper::~SundialsStepper()
{
	if( ERKStepMem != nullptr )
		ERKStepFree( &ERKStepMem );
	if( gInternal != nullptr )
		delete gInternal;
}


void SundialsStepper::Initialise( MomentsG** g0, double t0 )
{
	if( ERKStepMem != nullptr ) {
		throw std::runtime_error("Double Initialisation of Sundials Tiemstepper. Aborting.");
	}
	
	// This creates a GXVector that is a view of the data in the MomentsG** but 
	// does not *own* the data. Thus deleting this pointer will not free the underlying MomentsG
	gInternal = new GXVector( g0, ctx );

	// Wrap GXVector in an NVector

	gInternalNV = gInternal->asNVector();

	ERKStepMem = ERKStepCreate( SundialsStepper::SundialsF, t0, gInternalNV, ctx );
	if( ERKStepMem == nullptr )
		throw std::runtime_error("Unable to allocate SUNDIALS Memory. ABORT.");

	int retval;

	retval = ERKStepSStolerances( ERKStepMem, reltol, abstol );
	if( retval != ARK_SUCCESS ) {
		throw std::runtime_error("Internal SUNDIALS Error in ERKStepSStolerances.");
	}
	
	// Use the Shu-Osher 3rd order SSP method, with embedding
	retval = ERKStepSetTableNum( ERKStepMem, ARKODE_SHU_OSHER_3_2_3 );

	if( retval != ARK_SUCCESS ) {
		throw std::runtime_error("Internal SUNDIALS Error in ERKStepSetTableNum.");
	}

	ERKStepSetUserData( ERKStepMem, static_cast<void*>(this) );
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

	retval = ERKStepEvolve( ERKStepMem, t_out, gInternalNV, t );

	if( retval != ARK_SUCCESS ) {
		throw runtime_error("Error in ERKStepEvolve.");
	}
}

int SundialsStepper::SundialsF( sunrealtype t, N_Vector y, N_Vector ydot, void* userdata )
{
	GXVector *g = reinterpret_cast<GXVector*>( y->content );
	GXVector *gdot = reinterpret_cast<GXVector*>( ydot->content );
	return reinterpret_cast<SundialsStepper*>( userdata )->SundialsRHS( t, g, gdot );
}

int SundialsStepper::SundialsRHS( double time, GXVector const *g, GXVector * gdot )
{
	// Make sure fields are evaluated at this current g
	solver_->fieldSolve( *g, fields_ );

	// compute nonlinear term
	gdot->SetZero();

	if (nonlinear_ != nullptr) {
		for( int is = 0; is < grids_->Nspecies; ++is ) {
			nonlinear_->nlps ( (*g)[is], fields_, (*gdot)[is]);
		}
	}

	// compute and accumulate linear term
	for( int i = 0; i < grids_->Nspecies; ++i)
		linear_->rhs( (*g)[is], fields_, (*gdot)[is], dt_ );

}

