#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "linear.h"
#include "nonlinear.h"
#include "solver.h"
#include "forcing.h"
#include "grad_parallel.h"
#include "exb.h"

#include <sundials/sundials_context.hpp>
#include <arkode/arkode_butcher.h>
#include "gxvector.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual void advance(double* t, MomentsG** G, Fields* fields) = 0;
  virtual double get_dt() = 0;
};

class RungeKutta2 : public Timestepper {
 public:
  RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~RungeKutta2();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** G, MomentsG* GRhs,
		 Fields* f, double adt, bool setdt);

  double dt_;
  const double dt_max;
  const double cfl_fac = 1.0;
  double omega_max[3];
  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG   * GRhs       ;
  MomentsG  ** G1         ;
};

class RungeKutta3 : public Timestepper {
 public:
  RungeKutta3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~RungeKutta3();
  void advance(double* t, MomentsG** G, Fields* fields);
  void partial(MomentsG** G, MomentsG** Gt, Fields *f,
	       MomentsG** Rhs, MomentsG **Gnew, double adt, bool setdt);
  double get_dt() {return dt_;};

 private:
  const double dt_max;
  double dt_;
  const double cfl_fac = 1.73;
  double omega_max[3];

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG  ** GRhs1      ;
  MomentsG  ** GRhs2      ;
  MomentsG  ** G_q1       ;
  MomentsG  ** G_q2       ;
};

class RungeKutta4 : public Timestepper {
 public:
  RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~RungeKutta4();
  void advance(double* t, MomentsG** G, Fields* fields);
  void partial(MomentsG** G, MomentsG** Gt, Fields *f,
	       MomentsG** Rhs, MomentsG **Gnew, double adt, bool setdt);
  double get_dt() {return dt_;};

 private:
  const double dt_max;
  double dt_;
  const double cfl_fac = 2.82;
  double omega_max[3];

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG  ** GStar      ;
  MomentsG  ** GRhs       ;
  MomentsG  ** G_q1       ;
  MomentsG  ** G_q2       ;
};

class Ketcheson10 : public Timestepper {
 public:
  Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~Ketcheson10();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG** G_q1, MomentsG** GRhs, MomentsG* Gtmp, Fields* f,  bool setdt);
  const double dt_max;
  double dt_;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  GradParallel * grad_par   ;
  Forcing      * forcing_   ;
  ExB          * exb_       ;
  MomentsG    ** G_q1       ;
  MomentsG    ** G_q2       ;
  MomentsG     * Gtmp       ;
};

class K2 : public Timestepper {
 public:
  K2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~K2();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG** G_q1, MomentsG** GRhs, Fields* f, bool setdt);
  void FinalStep(MomentsG** G_q1, MomentsG** G_q2, MomentsG** GRhs, Fields* f);
  const double dt_max;
  double dt_;
  int stages_;
  double sm1inv;
  double sinv;
  
  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG  ** G_q1       ;
  MomentsG  ** G_q2       ;
};

class SSPx2 : public Timestepper {
 public:
  SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~SSPx2();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG* GRhs, Fields* f, bool setdt);
  const double dt_max;
  double dt_;
  double omega_max[3];
  const double adt = 1./sqrt(2.);
  const double cfl_fac = 1.0;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG  ** G1         ;
  MomentsG  ** G2         ;
  MomentsG   * GRhs       ;
};

class SSPx3 : public Timestepper {
 public:
  SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~SSPx3();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG* GRhs, Fields* f, bool setdt);

  const double dt_max;
  double omega_max[3];
  const double adt = pow(1./6., 1./3.);
  const double wgtfac = sqrt(9. - 2.* pow(6.,2./3.));
  const double w1 = 0.5 * (wgtfac - 1.);
  const double w2 = 0.5 * (pow(6.,2./3.) - 1 - wgtfac);
  const double w3 = 1./adt - 1. - w2*(w1+1.);
  const double cfl_fac = 1.73;
 
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  ExB          * exb_       ;
  GradParallel * grad_par   ;
  MomentsG    ** G1         ;
  MomentsG    ** G2         ;
  MomentsG    ** G3         ;
  MomentsG    *  GRhs       ;
  double dt_;
};

class G3 : public Timestepper {
 public:
  G3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~G3();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG** G_q1, MomentsG** GRhs, Fields* f,  bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  ExB        * exb_       ;
  MomentsG  ** G_u1       ;
  MomentsG  ** G_u2       ;
};

class GXVRK4 : public Timestepper {
 public:
  GXVRK4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in);
  ~GXVRK4();
  void advance(double* t, MomentsG** G, Fields* fields);
  void partial(GXVector & G, GXVector & Gt, Fields *f, 
		         GXVector & Rhs, GXVector & Gnew, double adt, bool setdt);
  double get_dt() {return dt_;};

 private:
  sundials::Context ctx;
  const double dt_max;
  double dt_;
  const double cfl_fac = 2.82;
  double omega_max[3];

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  ExB        * exb_       ;
  Forcing    * forcing_   ;
  Fields	    * fields_    ;
  GXVector GStar;
  GXVector GRhs;
  GXVector G_q1;
  GXVector G_q2;
};

class SundialsStepper : public Timestepper {
 public:
  SundialsStepper(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, ExB *exb, double dt_in, MomentsG** G0 );
  ~SundialsStepper();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

  // Wrapper around the rhs of dy/dt = F(y,t)
  static int SundialsF( sunrealtype t, N_Vector y, N_Vector ydot, void* data );
  int SundialsRHS( double t, GXVector * g, GXVector* gDot );

 private:
  void Initialise( MomentsG**, double );
  sundials::Context ctx;
  void *ERKStepMem;
  GXVector *gInternal, *gTmp;
  N_Vector gInternalNV;

  const double dt_;
  const double reltol = 1e-3;
  const double abstol = 1e-3;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  ExB        * exb_       ;
  Forcing    * forcing_   ;
  Fields	    * fields_    ;
};

