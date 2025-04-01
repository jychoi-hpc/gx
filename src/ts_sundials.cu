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
        fprintf(stderr, "SUNDIALS ARKODE Error: %s (code=%d)  \"%s\" at %s:%d \n", ARKodeGetReturnFlagName(code), (unsigned int)code, func, file, line);
        throw std::runtime_error("Error in SUNDIALS AKRODE. Aborting");
    }
}

// ============= Sundials-based Timestepping =============
SundialsStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
        Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** g0, double t0 ) :
    linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
    forcing_(forcing), exb_(exb), dt_(dt_in), ctx(), ARKodeMem(nullptr), gInternal(nullptr), fields_(nullptr)
{
    if( ARKodeMem != nullptr ) {
        throw std::runtime_error("Double Initialisation of Sundials Timestepper. Aborting.");
    }

    // This creates a GXVector that is a view of the data in the MomentsG** but 
    // does not *own* the data. Thus deleting this pointer will not free the underlying MomentsG
    gInternal = new GXVector( g0, ctx );

    // This is the clone constructor, allocates new RAM
    gTmp = new GXVector( *gInternal );

    // Wrap GXVector in an NVector

    gInternalNV = gInternal->asNVector();

    std::cout << "Initialising SUNDIALS ARKode library for timestepping." << std::endl;

    ARKodeMem = ERKStepCreate( SundialsStepper::SundialsF, t0, gInternalNV, ctx );

    if( ARKodeMem == nullptr ) {
        throw std::runtime_error("Unable to allocate SUNDIALS Memory. ABORT.");
    }

    // Load from parameters
    reltol = pars->SundialsRelTol;
    abstol = pars->SundialsAbsTol;

    ARKODECheck( ARKodeWFtolerances( ARKodeMem, SundialsStepper::SundialsErrorWeights ) );

    if( pars_->SundialsExplicitOrder > 0 ) { // If order is specified, this overrides a specific name
        ARKODECheck( ARKodeSetOrder( ARKodeMem, pars_->SundialsExplicitOrder ) );
    } else { // If order not specified, choose by name, which has a default
        ARKODECheck( ERKStepSetTableName( ARKodeMem, pars_->SundialsExplicitScheme.c_str() ) );
    }

    ARKODECheck( ARKodeSetMinStep( ARKodeMem, pars_->SundialsMinStep ) );

    ARKODECheck( ARKodeSetMaxStep( ARKodeMem, pars_->SundialsMaxStep ) );

    ARKODECheck( ARKodeSetUserData( ARKodeMem, static_cast<void*>(this) ) );

    ARKODECheck( ARKodeSetInitStep( ARKodeMem, dt_ ) );

    ARKODECheck( ARKodeSetMaxNumSteps( ARKodeMem, pars_->SundialsMaxInternalSteps ) );

    if( pars_->SundialsFixedTimestep ) {
        ARKODECheck( ARKodeSetFixedStep( ARKodeMem, dt_ ) );
    }

    if( pars_->cfl > 0.0 ) {
        ARKODECheck( ARKodeSetCFLFraction( ARKodeMem, pars_->cfl ) );
    }

    
    // By default be more relaxed in the last 25% in every grid dimension
    unsigned int ky_max = (grids_->Ny - 1)/3 + 1;
    unsigned int kx_max = (grids_->Nx - 1)/3 + 1;
    setWeightingConstants( 
            std::ceil( 0.75 * grids_->Nm ),
            std::ceil( 0.75 * grids_->Nl ),
            std::ceil( 0.75 * kx_max ),
            std::ceil( 0.75 * ky_max ),
            std::sqrt( reltol ) );


}

SundialsStepper::~SundialsStepper()
{
    if( ARKodeMem != nullptr )
        ARKodeFree( &ARKodeMem );
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

    retval = ARKodeEvolve( ARKodeMem, t_out, gInternalNV, &time, ARK_NORMAL );

    if( retval != ARK_SUCCESS ) {
        throw std::runtime_error("Error in ARKodeEvolve.");
    }

    *t = time;

    if( fabsf( *t - t_out ) > 1e-3 )
    {
        throw std::runtime_error("Unable to advance to requested time " + std::to_string(t_out) + " reached " + std::to_string(*t) + " aborting.");
    }

    // Set fields_ to be the fields consistent with the final state (for diagnostics etc)
    solver_->fieldSolve( *gInternal, fields_ );
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

    for( size_t i = 0; i < g->nSpecies(); ++i )
    {
        MomentsG const & m = *((*g)[ i ]);
        float max_abs_g = g->MaxNorm(); // obtain Max |g| so we can ignore tolerances on modes with |g_i| < 1e-3 * Max(|g|)
        setWeightsKernel<<< m.dG_all, m.dB_all >>> ( weights->gData( i ), m, abstol, reltol, wg_tol, max_abs_g );

        checkCuda( cudaGetLastError() );
    }

    return 0;
}

double SundialsStepper::get_dt() 
{
    double step;
    int retval = ARKodeGetCurrentStep( ARKodeMem, &step );
    if( retval == ARK_SUCCESS ) {
        return step;
    } else {
        throw std::runtime_error("Error Encountered in ARKodeGetCurrentStep");
        return -1.0;
    }
}

long int SundialsStepper::getRHSEvals()
{
    long int nRHS = 0;
    ARKODECheck( ERKStepGetNumRhsEvals( ARKodeMem, &nRHS ) );
    return nRHS;
}

long int SundialsStepper::getNSteps()
{
    long int nSteps = 0;
    ARKODECheck( ARKodeGetNumSteps( ARKodeMem, &nSteps ) );
    return nSteps;
}

// Fixed implementations of non-adaptive RK4 / SSPX2 / SSPX3 / K10 to allow for seamless transition
sunrealtype SunRK4Stepper::c[4] = {0.0,0.5,0.5,1.0};
sunrealtype SunRK4Stepper::b[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
sunrealtype SunRK4Stepper::a[16] = {0.0,0.0,0.0,0.0,
                                    0.5,0.0,0.0,0.0,
                                    0.0,0.5,0.0,0.0,
                                    0.0,0.0,1.0,0.0};

SunRK4Stepper::SunRK4Stepper(Linear *linear, Nonlinear *nonlinear, Solver *solver, Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** G0, double t0 ) 
    : SundialsStepper( linear, nonlinear, solver, pars, grids, forcing, exb, dt_in, G0, t0 ), rk4table( nullptr )
{
   rk4table = ARKodeButcherTable_Create( 4, 4, 0, c, a, b, nullptr ); 
   if( rk4table == nullptr )
       throw std::runtime_error("Could not create RK4 ButcherTable!");

   ARKODECheck( ERKStepSetTable( ARKodeMem, rk4table ) );

   ARKODECheck( ARKodeSetFixedStep( ARKodeMem, dt_ ) );
}

SunRK4Stepper::~SunRK4Stepper()
{
    ARKodeButcherTable_Free( rk4table );
}


// We rely on the fact that the MomentsG is the same one we were told about initially
void SunRK4Stepper::advance(double *t, MomentsG** , Fields* f)
{
    double t_out = *t + dt_; // Advance one time step to t_out

    int retval;

    fields_ = f;

    // Needed for when sunrealtype is float not double
    sunrealtype time = *t;

    retval = ARKodeEvolve( ARKodeMem, t_out, gInternalNV, &time, ARK_ONE_STEP );

    if( retval != ARK_SUCCESS ) {
        throw std::runtime_error("Error in ARKodeEvolve.");
    }

    *t = time;

    if( fabsf( *t - t_out ) > 1e-3 )
    {
        throw std::runtime_error("Unable to advance to requested time " + std::to_string(t_out) + " reached " + std::to_string(*t) + " aborting.");
    }

    // Set fields_ to be the fields consistent with the final state (for diagnostics etc)
    solver_->fieldSolve( *gInternal, fields_ );
    checkCuda( cudaGetLastError() );
}

// ============= Sundials Low Memory Timestepping =============
SundialsLSRKStepper::SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
        Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** g0, double t0 ) :
    linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
    forcing_(forcing), exb_(exb), dt_(dt_in), ctx(), ARKodeMem(nullptr), gInternal(nullptr), fields_(nullptr)
{
    if( ARKodeMem != nullptr ) {
        throw std::runtime_error("Double Initialisation of Sundials Timestepper. Aborting.");
    }

    // This creates a GXVector that is a view of the data in the MomentsG** but 
    // does not *own* the data. Thus deleting this pointer will not free the underlying MomentsG
    gInternal = new GXVector( g0, ctx );

    // This is the clone constructor, allocates new RAM
    gTmp = new GXVector( *gInternal );

    // Wrap GXVector in an NVector

    gInternalNV = gInternal->asNVector();

    std::cout << "Initialising SUNDIALS ARKode library for timestepping." << std::endl;

    ARKodeMem = ERKStepCreate( SundialsLSRKStepper::SundialsF, t0, gInternalNV, ctx );

    if( ARKodeMem == nullptr ) {
        throw std::runtime_error("Unable to allocate SUNDIALS Memory. ABORT.");
    }

    // Load from parameters
    reltol = pars->SundialsRelTol;
    abstol = pars->SundialsAbsTol;

    ARKODECheck( ARKodeWFtolerances( ARKodeMem, SundialsLSRKStepper::SundialsErrorWeights ) );

    if( pars_->SundialsExplicitOrder > 0 ) { // If order is specified, this overrides a specific name
        ARKODECheck( ARKodeSetOrder( ARKodeMem, pars_->SundialsExplicitOrder ) );
    } else { // If order not specified, choose by name, which has a default
        ARKODECheck( ERKStepSetTableName( ARKodeMem, pars_->SundialsExplicitScheme.c_str() ) );
    }

    ARKODECheck( ARKodeSetMinStep( ARKodeMem, pars_->SundialsMinStep ) );

    ARKODECheck( ARKodeSetMaxStep( ARKodeMem, pars_->SundialsMaxStep ) );

    ARKODECheck( ARKodeSetUserData( ARKodeMem, static_cast<void*>(this) ) );

    ARKODECheck( ARKodeSetInitStep( ARKodeMem, dt_ ) );

    ARKODECheck( ARKodeSetMaxNumSteps( ARKodeMem, pars_->SundialsMaxInternalSteps ) );

    if( pars_->SundialsFixedTimestep ) {
        ARKODECheck( ARKodeSetFixedStep( ARKodeMem, dt_ ) );
    }

    if( pars_->cfl > 0.0 ) {
        ARKODECheck( ARKodeSetCFLFraction( ARKodeMem, pars_->cfl ) );
    }

    
    // By default be more relaxed in the last 25% in every grid dimension
    unsigned int ky_max = (grids_->Ny - 1)/3 + 1;
    unsigned int kx_max = (grids_->Nx - 1)/3 + 1;
    setWeightingConstants( 
            std::ceil( 0.75 * grids_->Nm ),
            std::ceil( 0.75 * grids_->Nl ),
            std::ceil( 0.75 * kx_max ),
            std::ceil( 0.75 * ky_max ),
            std::sqrt( reltol ) );


}

SundialsLSRKStepper::~SundialsStepper()
{
    if( ARKodeMem != nullptr )
        ARKodeFree( &ARKodeMem );
    if( gInternal != nullptr )
        delete gInternal;
    if( gTmp != nullptr )
        delete gTmp;
}

// We rely on the fact that the MomentsG is the same one we were told about initially
void SundialsLSRKStepper::advance(double *t, MomentsG** , Fields* f)
{
    double t_out = *t + dt_; // Try to advance one `time step', possibly using multiple internal steps

    int retval;

    fields_ = f;

    // Needed for when sunrealtype is float not double
    sunrealtype time = *t;

    retval = ARKodeEvolve( ARKodeMem, t_out, gInternalNV, &time, ARK_NORMAL );

    if( retval != ARK_SUCCESS ) {
        throw std::runtime_error("Error in ARKodeEvolve.");
    }

    *t = time;

    if( fabsf( *t - t_out ) > 1e-3 )
    {
        throw std::runtime_error("Unable to advance to requested time " + std::to_string(t_out) + " reached " + std::to_string(*t) + " aborting.");
    }

    // Set fields_ to be the fields consistent with the final state (for diagnostics etc)
    solver_->fieldSolve( *gInternal, fields_ );
    checkCuda( cudaGetLastError() );
}

int SundialsLSRKStepper::SundialsF( sunrealtype t, N_Vector y, N_Vector ydot, void* userdata )
{
    GXVector *g = reinterpret_cast<GXVector*>( y->content );
    GXVector *gdot = reinterpret_cast<GXVector*>( ydot->content );
    int retval = reinterpret_cast<SundialsStepper*>( userdata )->SundialsRHS( t, g, gdot );
    return retval;
}

int SundialsLSRKStepper::SundialsRHS( double time, GXVector *g, GXVector * gdot )
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


int SundialsLSRKStepper::SundialsErrorWeights( N_Vector y, N_Vector ewt, void * userdata )
{
    GXVector* g       = reinterpret_cast<GXVector*>( y->content   );
    GXVector* weights = reinterpret_cast<GXVector*>( ewt->content );
    return reinterpret_cast<SundialsStepper*>( userdata )->ErrorWeights( g, weights );
}

int SundialsLSRKStepper::ErrorWeights( GXVector *g, GXVector *weights )
{
    // Just do the usual abstol / reltol stuff
    // weights[i] = 1/(abstol + reltol*|g[i]|)
    // but with one fused kernel to avoid multiple passes over the data

    for( size_t i = 0; i < g->nSpecies(); ++i )
    {
        MomentsG const & m = *((*g)[ i ]);
        float max_abs_g = g->MaxNorm(); // obtain Max |g| so we can ignore tolerances on modes with |g_i| < 1e-3 * Max(|g|)
        setWeightsKernel<<< m.dG_all, m.dB_all >>> ( weights->gData( i ), m, abstol, reltol, wg_tol, max_abs_g );

        checkCuda( cudaGetLastError() );
    }

    return 0;
}

double SundialsLSRKStepper::get_dt() 
{
    double step;
    int retval = ARKodeGetCurrentStep( ARKodeMem, &step );
    if( retval == ARK_SUCCESS ) {
        return step;
    } else {
        throw std::runtime_error("Error Encountered in ARKodeGetCurrentStep");
        return -1.0;
    }
}

long int SundialsLSRKStepper::getRHSEvals()
{
    long int nRHS = 0;
    ARKODECheck( ERKStepGetNumRhsEvals( ARKodeMem, &nRHS ) );
    return nRHS;
}

long int SundialsLSRKStepper::getNSteps()
{
    long int nSteps = 0;
    ARKODECheck( ARKodeGetNumSteps( ARKodeMem, &nSteps ) );
    return nSteps;
}
