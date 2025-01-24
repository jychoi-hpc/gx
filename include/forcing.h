#pragma once
#include "gpu_defs.h"
#include "moments.h"
#include "parameters.h"
#include "device_funcs.h"

class Forcing {
  public:
   virtual ~Forcing() {};
   virtual void stir(MomentsG *G) = 0;
};

class KzForcing : public Forcing {
 public:
  KzForcing(Parameters *pars);
  ~KzForcing();
  void stir(MomentsG *G);

 protected:
  float forcing_amp_;
  cuComplex rf;
  Parameters * pars_ ;

};
class HeliInjForcing : public Forcing {
 public:
  HeliInjForcing(Parameters *pars, Grids *grids);
  ~HeliInjForcing();
  void stir(MomentsG *G);
 protected:
  float pos_forcing_amp_;
  float neg_forcing_amp_;
  int k2min;
  int k2max;
  int kx;
  int ky;
  int mag_squared;
  int kz;
  int Nz;
  cuComplex rf;
  Parameters * pars_ ;
  Grids * grids_;
};
class KzForcingImpulse : public KzForcing {
 public:
  KzForcingImpulse(Parameters *pars);
    void stir(MomentsG *G);

    private:
    bool stirring_done;
};

class genForcing : public Forcing {
 public:
  genForcing(Parameters *pars);
  ~genForcing();
  void stir(MomentsG *G);

 protected:
  float forcing_amp_;
  cuComplex rf;
  Parameters * pars_ ;
};


