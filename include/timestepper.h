#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "linear.h"
#include "nonlinear.h"
#include "solver.h"
#include "forcing.h"
#include "grad_parallel.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual void advance(double* t, MomentsG* G, Fields* fields) = 0;
  virtual double get_dt() = 0;
};

// ================================
// Classical Runge-Kutta methods
// see e.g. Durran Ch 2.3.1-2.3.2
// ================================

// classic second-order RK2 
class RungeKutta2 : public Timestepper {
 public:
  RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta2();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* G, MomentsG* GRhs,
		 Fields* f, double adt, bool setdt);

  double dt_;
  const double dt_max;
  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  MomentsG   * GRhs       ;
  MomentsG   * G1         ;
};

// classic fourth-order RK4 
class RungeKutta4 : public Timestepper {
 public:
  RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta4();
  void advance(double* t, MomentsG* G, Fields* fields);
  void partial(MomentsG* G, MomentsG* Gt, Fields *f,
	       MomentsG* Rhs, MomentsG *Gnew, double adt, bool setdt);
  double get_dt() {return dt_;};

 private:
  const double dt_max;
  double dt_;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  MomentsG   * GStar      ;
  MomentsG   * GRhs       ;
  MomentsG   * G_q1       ;
  MomentsG   * G_q2       ;
};

// ================================
// SSP Runge-Kutta methods
// see e.g. Durran Ch 2.3.3
// ================================

// 3-stage 3rd order SSP-RK scheme of Shu & Osher (1988)
class SSPRK3 : public Timestepper {
 public:
  SSPRK3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPRK3();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);

  const double dt_max;
 
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     * G1         ;
  MomentsG     * G2         ;
  MomentsG     * GRhs       ;
  double dt_;
};

class IMEX_SSPRK3_DIRK : public Timestepper {
 public:
  IMEX_SSPRK3_DIRK(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~IMEX_SSPRK3_DIRK();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

  void explicit_terms(MomentsG* G1, MomentsG* G, Fields* f, bool setdt);
  void implicit_terms(MomentsG* G1, MomentsG* G, Fields* f);
  void invert_implicit_terms(MomentsG* G1, double rdt);

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);

  const double dt_max;
 
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     * G1         ;
  MomentsG     * G2         ;
  MomentsG     * A0         ;
  MomentsG     * A1         ;
  MomentsG     * A2         ;
  MomentsG     * B0         ;
  MomentsG     * B1         ;
  MomentsG     * B2         ;
  double dt_;
};

// low-storage 10-stage 4th order SSP-RK method (Ketcheson, SIAM JSC 2008)
class Ketcheson10 : public Timestepper {
 public:
  Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~Ketcheson10();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f,  bool setdt);
  const double dt_max;
  double dt_;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  GradParallel * grad_par   ;
  Forcing      * forcing_   ;
  MomentsG     * G_q1       ;
  MomentsG     * G_q2       ;
};

class K2 : public Timestepper {
 public:
  K2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~K2();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, bool setdt);
  void FinalStep(MomentsG* G_q1, MomentsG* G_q2, MomentsG* GRhs, Fields* f);
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

  MomentsG   * G_q1       ;
  MomentsG   * G_q2       ;
};

// ================================
// extended SSP Runge-Kutta methods
// unpublished, but from G. Hammett
// ================================

// extended SSP-RK2
class SSPx2 : public Timestepper {
 public:
  SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPx2();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  MomentsG   * G1         ;
  MomentsG   * G2         ;
  MomentsG   * GRhs       ;
};

// extended SSP-RK3
class SSPx3 : public Timestepper {
 public:
  SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPx3();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);

  const double dt_max;
  const double adt = pow(1./6., 1./3.);
  const double wgtfac = sqrt(9. - 2.* pow(6.,2./3.));
  const double w1 = 0.5 * (wgtfac - 1.);
  const double w2 = 0.5 * (pow(6.,2./3.) - 1 - wgtfac);
  const double w3 = 1./adt - 1. - w2*(w1+1.);
 
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     * G1         ;
  MomentsG     * G2         ;
  MomentsG     * G3         ;
  MomentsG     * GRhs       ;
  double dt_;
};

class G3 : public Timestepper {
 public:
  G3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~G3();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f,  bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    ;
  Nonlinear  * nonlinear_ ;
  Solver     * solver_    ;
  Parameters * pars_      ;
  Grids      * grids_     ;
  Forcing    * forcing_   ;
  MomentsG   * G_u1       ;
  MomentsG   * G_u2       ;
};

/*
class SDCe : public Timestepper {
 public:
  SDCe(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SDCe();
  void advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  void full_rhs(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar);
  Linear *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids *grids_;
  Forcing *forcing_;
  const double dt_max;
  
  double dt_;
}
*/
