#pragma once
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "grids.h"
#include "species.h"
#include <cublas_v2.h>

class CollisionOperator {
 public:
  virtual ~CollisionOperator() {};
  virtual void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate) = 0;
};

class LorentzCollisionOperator : public CollisionOperator {
 public:
  LorentzCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob);
  ~LorentzCollisionOperator();
  void applyCollisionMatrix(cuComplex* G_in, cuComplex* G_res, bool accumulate);
  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate);

 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  cuComplex* collMat{};
  cublasHandle_t handle;

  MomentsG* tmpG{};

  const int Nlm;
  const int Nk;
  const specie sp_a;
  const int is_a;
  dim3 dimGrid, dimBlock;

  void initCollisionMatrix(cuComplex* mat);
};
