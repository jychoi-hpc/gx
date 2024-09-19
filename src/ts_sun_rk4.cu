
sunrealtype SunRK4Stepper::c{0.0,0.5,0.5,1.0};
sunrealtype SunRK4Stepper::b{1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
sunrealtype SunRK4Stepper::a{0.0,0.0,0.0,0.0,
                             0.5,0.0,0.0,0.0,
                             0.0,0.5,0.0,0.0,
                             0.0,0.0,1.0,0.0};

SunRK4Stepper::SunRK4Stepper(Linear *linear, Nonlinear *nonlinear, Solver *solver, Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** G0, double t0 ) 
    : SundialsStepper( linear, nonlinear, solver, pars, grids, forcing, exb, dt_in, G0, t0 ), rk4table( nullptr )
{
   rk4table = ARKodeButcherTable_Create( 4, 4, 0, c, a, b, nullptr ); 
   if( rk4table == nullptr )
       throw std::runtime_error("Could not create RK4 ButcherTable!");

   ARKodeCheck( ERKStepSetTable( ERKStepMem, rk4table ) );

   ARKODECheck( ERKStepSetFixedStep( ERKStepMem, dt_ ) );
}

SunRK4Stepper::~SunRK4Stepper()
{
    ARKodeButcherTable_Free( rk4table );
}

