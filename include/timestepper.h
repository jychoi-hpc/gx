#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "linear.h"
#include "nonlinear.h"
#include "solver.h"
#include "forcing.h"
#include "grad_parallel.h"
#include "cublas_test.h"
#include "cusolve.h"
#include "green.h"
#include "exb.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual void advance(double* t, MomentsG** G, Fields* fields) = 0;
  virtual double get_dt() = 0;
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
	       MomentsG** Rhs, MomentsG **Gnew, double adt );
  double get_dt() {return dt_;};

  void set_timestep( Fields * );

 private:
  const double dt_max;
  double dt_;
  const double cfl_fac = 2.82;
  double omega_max[3];
  bool set_dt = true;

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
  double omega_max[3];

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

// 3-stage 3rd order SSP-RK scheme of Shu & Osher (1988)
class SSPRK3 : public Timestepper {
 public:
  SSPRK3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPRK3();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG* GRhs, Fields* f, bool setdt);
  const double dt_max;
 
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     ** G1         ;
  MomentsG     ** G2         ;
  MomentsG      * GRhs       ;
  double dt_;
};


class Lie_Trotter : public Timestepper {
 public:
  Lie_Trotter(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv, const float* kperp2);
  ~Lie_Trotter();
  void advance(double* t, MomentsG** G, Fields* fields);
//  double get_dt() {return dt_;};
  double get_dt();
  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);

  void invert_implicit_terms_linked_lw(MomentsG** G1, MomentsG** G0, cuComplex** G_sm_s_phi, cuComplex** G_sm_s_apar, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron, bool flip);

  void invert_bounce(MomentsG** G1, cuComplex** Gc, cuComplex** Gr, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron, bool flip);

  void invert_streaming(MomentsG** G1, cuComplex** Gc, cuComplex** Gr, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron, bool flip);


  void ssprk3(MomentsG** A1, MomentsG** A2, MomentsG** A3, MomentsG** G, MomentsG** G1, Fields* f, bool setdt);

  double a21, a31, a32, w1, w2, w3;
  double p_, q_, r_, s_, t_, u_;
  bool sdirk;


 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** GRhs, Fields* f, bool setdt);
  const double dt_max;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Geometry     * geo_       ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Cublas_test  * cublas_    ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     ** G1         ;
  cuComplex    ** Gc         ;
  cuComplex    ** Gr         ;
  MomentsG     ** G0         ;
  MomentsG     ** G2         ;
  MomentsG     ** A1         ;
  MomentsG     ** A2         ;
  MomentsG     ** A3         ;
  MomentsG     ** B1         ;
  MomentsG     ** B2         ;
  MomentsG     ** B3         ;
  Cublas_test  ** mirror     ;
//  Cusolve      ** mirror    ;
  Fields	* f1	     ;
  cuComplex    ** phi_l      ;
  cuComplex    ** apar_l      ;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  const float gradpar_;
  const float* bmagInv_;
  const float* kperp2_;
  bool flip;
  dim3 dG, dB, dG_lw, dB_lw, dG_m1, dB_m1, dG_m2, dB_m2;
};




class IMEX_3stage_Full : public Timestepper {
 public:
  IMEX_3stage_Full(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv, const float* kperp2);
  ~IMEX_3stage_Full();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);
  void implicit_terms_id(MomentsG** G1, MomentsG** G, Fields* f);
  void apply_preconditioner(MomentsG** G1, MomentsG** G, Fields* f);

  void invert_implicit_terms(MomentsG** G1, cuComplex** G_sm_s_phi, cuComplex** G_sm_s_apar, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron);
  void invert_implicit_terms_linked(MomentsG** G1, cuComplex** G_sm_s_phi, cuComplex** G_sm_s_apar, MomentsG** G0, MomentsG** G2, Fields *f, cuComplex** phi_l, cuComplex** apar_l, double sdt,const float gradpar_, const float* bmagInv_, int ielectron);
  void invert_implicit_terms_linked_lw(MomentsG** G1, MomentsG** G0, Fields *f, double sdt,const float gradpar_, int ielectron);

  double a21, a31, a32, a41, a42, a43, a44, w1, w2, w3, w4;
  double p_, q_, r_, s_, t_, u_, w_, x_, y_, z_;
  bool sdirk;
  int nstages;


 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** GRhs, Fields* f, bool setdt);
  const double dt_max;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Geometry     * geo_       ;
  Cublas_test  ** mirror    ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     ** G1         ;
  cuComplex    ** G_sm_s_phi ;
  cuComplex    ** G_sm_s_apar;
  cuComplex    ** G_sm_b_apar;

  MomentsG     ** G0         ;
  MomentsG     ** G2         ;
  MomentsG     ** G3	     ;
  MomentsG     ** G4	     ;
  MomentsG     ** A1         ;
  MomentsG     ** A2         ;
  MomentsG     ** A3         ;
  MomentsG     ** B1         ;
  MomentsG     ** B2         ;
  MomentsG     ** B3         ;

  MomentsG     ** A4         ;
  MomentsG     ** B4         ;

  Fields	* f1	     ;
  cuComplex    ** phi_flr      ;
  cuComplex    ** apar_flr      ;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  const float gradpar_;
  const float* bmagInv_;
  const float* kperp2_;
  dim3 dG, dB, dG_lw, dB_lw, dG_m1, dB_m1, dG_m2, dB_m2, dG_all, dB_all;

};


class IMEX_3stage : public Timestepper {
 public:
  IMEX_3stage(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv);
  ~IMEX_3stage();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);
  void invert_implicit_terms(MomentsG** G1, MomentsG* Gc, MomentsG** Gr, Fields *f, double rdt, const float gradpar, const float* bmagInv, int ielectron);
 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** GRhs, Fields* f, bool setdt);
  const double dt_max;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     ** G1         ;
  MomentsG     ** Gc         ;
  MomentsG     ** Gr         ;
  MomentsG     ** A1         ;
  MomentsG     ** A2         ;
  MomentsG     ** A3         ;
  MomentsG     ** B1         ;
  MomentsG     ** B2         ;
  MomentsG     ** B3         ;
  Fields	* f1	     ;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  const float gradpar_;
  const float* bmagInv_;
  dim3 dG, dB;
};

class IMEX_4stage : public Timestepper {
 public:
  IMEX_4stage(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~IMEX_4stage();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};

  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);
  void invert_implicit_terms(MomentsG* G1, Fields* f, double rdt);

 private:
  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG    ** G1         ;
  MomentsG    ** A1         ;
  MomentsG    ** A2         ;
  MomentsG    ** A3         ;
  MomentsG    ** A4         ;
  MomentsG    ** B1         ;
  MomentsG    ** B2         ;
  MomentsG    ** B3         ;
  MomentsG    ** B4         ;
  Fields       * f1         ;
  const double dt_max;
  int nstage;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  dim3 dG, dB;
};

class G3 : public Timestepper {
 public:
  G3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
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
  MomentsG  ** G_u1       ;
  MomentsG  ** G_u2       ;
};

class Lie_Trotter_Green : public Timestepper {
 public:
  Lie_Trotter_Green(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Green *green, Geometry *geo, Forcing *forcing, double dt_in, const float gradpar, const float* bmagInv);
  ~Lie_Trotter_Green();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);
  void invert_implicit_terms(MomentsG** G, MomentsG** G1, Fields *f, double sdt,const float gradpar_, const float* kperp2, int ielectron);
  void invert_bounce_terms(MomentsG*G, int stage);
  void ssprk3(MomentsG** A1, MomentsG** A2, MomentsG** A3, MomentsG** G, MomentsG** G1, Fields* f, bool setdt);
  cuComplex     ** bounce_rhs;
  cuComplex     ** res;
  bool flip;
  cuComplex* phi_i;

 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** GRhs, Fields* f, bool setdt);
  const double dt_max;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Cusolve      * cusolve_   ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  Green        * green_     ;
  Geometry     * geo_       ;
  Cublas_test       * cublas_    ;
  MomentsG     ** G1         ;
  MomentsG     ** Gh         ;
  MomentsG     ** A1         ;
  MomentsG     ** A2         ;
  MomentsG     ** A3         ;
  MomentsG     ** B1         ;
  MomentsG     ** B2         ;
  MomentsG     ** B3         ;
  Fields	* f1	     ;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  const float gradpar_;
  const float* kperp2_;
  dim3 dG, dB, dG_b, dB_b, dG_l, dB_l;
};


class IMEX_3stage_Green : public Timestepper {
 public:
  IMEX_3stage_Green(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Green *green, Cublas_test *cublas, Forcing *forcing, double dt_in, const float gradpar, const float* kperp2);
  ~IMEX_3stage_Green();
  void advance(double* t, MomentsG** G, Fields* fields);
  double get_dt() {return dt_;};
  void explicit_terms(MomentsG** G1, MomentsG** G, Fields* f, bool setdt);
  void implicit_terms(MomentsG** G1, MomentsG** G, Fields* f);
  void invert_implicit_terms(MomentsG** G1, MomentsG** Gh, Fields *f, double rdt, const float gradpar, const float* kperp2, int ielectron);
  double a21, a31, a32, a41, a42, a43, a44, w1, w2, w3, w4;
  double p_, q_, r_, s_, t_, u_, w_, x_, y_, z_;
  bool sdirk;
  int nstages;
  cuComplex* phi_i;
  cuComplex* phi_copy;
  bool flip;

 private:
  void EulerStep(MomentsG** G1, MomentsG** G0, MomentsG** GRhs, Fields* f, bool setdt);
  const double dt_max;

  Linear       * linear_    ;
  Nonlinear    * nonlinear_ ;
  Solver       * solver_    ;
  Parameters   * pars_      ;
  Grids        * grids_     ;
  Green        * green_;
  Cublas_test  * cublas_    ;
  Forcing      * forcing_   ;
  GradParallel * grad_par   ;
  MomentsG     ** G1         ;
  MomentsG     ** Gh         ;
  MomentsG     ** A1         ;
  MomentsG     ** A2         ;
  MomentsG     ** A3         ;
  MomentsG     ** B1         ;
  MomentsG     ** B2         ;
  MomentsG     ** B3         ;

  MomentsG     ** A4         ;
  MomentsG     ** B4         ;
  Fields	* f1	     ;

  double dt_;
  int ielectron;
  double vte;
  double zte;
  const float gradpar_;
  const float* kperp2_;
  dim3 dG, dB, dG_b, dB_b, dG_l, dB_l;
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

