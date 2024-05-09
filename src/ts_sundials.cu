#include "timestepper.h"
#include "get_error.h"
#include <iostream>

#include <stdexcept>
#include <string>

#include <arkode/arkode_erkstep.h>

extern "C" {
#include <stdio.h>
#include <assert.h>
}

#define ARKODECheck(val)           __checkARKODEErrors__ ( (val), #val, __FILE__, __LINE__ )

template <typename T> inline void __checkARKODEErrors__(T code, const char *func, const char *file, int line) 
{
  if (code != ARK_SUCCESS) {
    fprintf(stderr, "SUNDIALS ARKODE Error: %s (code=%d)  \"%s\" at %s:%d \n", ERKStepGetReturnFlagName(code), (unsigned int)code, func, file, line);
	 throw std::runtime_error("Error in SUNDIALS AKRODE. Aborting");
  }
}


// ============= Sundials-based Timestepping =============
SundialsStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** g0, double t0 ) :
   linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
   forcing_(forcing), exb_(exb), dt_(dt_in), ctx(), ERKStepMem(nullptr), gInternal(nullptr), fields_(nullptr)
{
   if( ERKStepMem != nullptr ) {
      throw std::runtime_error("Double Initialisation of Sundials Timestepper. Aborting.");
   }

   // This creates a GXVector that is a view of the data in the MomentsG** but 
   // does not *own* the data. Thus deleting this pointer will not free the underlying MomentsG
   gInternal = new GXVector( g0, ctx );

   // This is the clone constructor, allocates new RAM
   gTmp = new GXVector( *gInternal );

   // Wrap GXVector in an NVector

   gInternalNV = gInternal->asNVector();

   std::cout << "Initialising SUNDIALS ERKStep library for timestepping." << std::endl;

   ERKStepMem = ERKStepCreate( SundialsStepper::SundialsF, t0, gInternalNV, ctx );

   if( ERKStepMem == nullptr ) {
      throw std::runtime_error("Unable to allocate SUNDIALS Memory. ABORT.");
	}

   int retval;

   // Load from parameters
   reltol = pars->SundialsRelTol;
   abstol = pars->SundialsAbsTol;

   // Apply
   ARKODECheck( ERKStepWFtolerances( ERKStepMem, SundialsStepper::SundialsErrorWeights ) );

   if( pars_->SundialsExplicitOrder > 0 ) { // If order is specified, this overrides a specific name
      ARKODECheck( ERKStepSetOrder( ERKStepMem, pars_->SundialsExplicitOrder ) );
   } else { // If order not specified, choose by name, which has a default
      ARKODECheck( ERKStepSetTableName( ERKStepMem, pars_->SundialsExplicitScheme.c_str() ) );
   }

   ARKODECheck( ERKStepSetMinStep( ERKStepMem, pars_->SundialsMinStep ) );

   ARKODECheck( ERKStepSetMaxStep( ERKStepMem, pars_->SundialsMaxStep ) );

   ARKODECheck( ERKStepSetUserData( ERKStepMem, static_cast<void*>(this) ) );

   ARKODECheck( ERKStepSetInitStep( ERKStepMem, dt_ ) );
   
	ARKODECheck( ERKStepSetMaxNumSteps( ERKStepMem, pars_->SundialsMaxInternalSteps ) );

   if( pars_->SundialsFixedTimestep ) {
      ARKODECheck( ERKStepSetFixedStep( ERKStepMem, dt_ ) );
   }

   if( pars_->cfl > 0.0 ) {
      ARKODECheck( ERKStepSetCFLFraction( ERKStepMem, pars_->cfl ) );
   }

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

// We rely on the fact that the MomentsG is the same one we were told about initially
void SundialsStepper::advance(double *t, MomentsG** , Fields* f)
{
   double t_out = *t + dt_; // Try to advance one `time step', possibly using multiple internal steps

   int retval;

   fields_ = f;

   // Needed for when sunrealtype is float not double
   sunrealtype time = *t;

   retval = ERKStepEvolve( ERKStepMem, t_out, gInternalNV, &time, ARK_NORMAL );

   if( retval != ARK_SUCCESS ) {
      throw std::runtime_error("Error in ERKStepEvolve.");
   }

   *t = time;

   if( fabsf( *t - t_out ) > 1e-3 )
   {
      throw std::runtime_error("Unable to advance to requested time " + std::to_string(t_out) + " reached " + std::to_string(*t) + " aborting.");
   }

   // Set fields_ to be the fields consistent with the final state (for diagnostics etc)
   solver_->fieldSolve( *gInternal, fields_ );
   checkCuda( cudaGetLastError() );
   checkCuda( cudaDeviceSynchronize() );
   checkCuda( cudaGetLastError() );
}

int SundialsStepper::SundialsF( sunrealtype t, N_Vector y, N_Vector ydot, void* userdata )
{
   GXVector *g = reinterpret_cast<GXVector*>( y->content );
   GXVector *gdot = reinterpret_cast<GXVector*>( ydot->content );
   int retval = reinterpret_cast<SundialsStepper*>( userdata )->SundialsRHS( t, g, gdot );
   return retval;
}

int SundialsStepper::SundialsRHS( double time, GXVector *g, GXVector * gdot )
{
   checkCuda( cudaGetLastError() );
   // Synchronise data from all nodes
   g->sync();

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

   checkCuda( cudaGetLastError() );

   return 0;
}


int SundialsStepper::SundialsErrorWeights( N_Vector y, N_Vector ewt, void * userdata )
{
   GXVector* g       = reinterpret_cast<GXVector*>( y->content   );
   GXVector* weights = reinterpret_cast<GXVector*>( ewt->content );
   return reinterpret_cast<SundialsStepper*>( userdata )->ErrorWeights( g, weights );
}

int SundialsStepper::ErrorWeights( GXVector *g, GXVector *weights )
{
   // Just do the usual abstol / reltol stuff
   // weights[i] = 1/(abstol + reltol*|g[i]|)
   // but with one fused kernel to avoid multiple passes over the data

   for( int i = 0; i < g->nSpecies(); ++i )
   {
      MomentsG const & m = *((*g)[ i ]);
      setWeightsKernel<<< m.dG_all, m.dB_all >>> ( weights->gData( i ), m, abstol, reltol );
      checkCuda( cudaGetLastError() );
   }

   return 0;
}
