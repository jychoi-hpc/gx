#pragma once
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "grids.h"
#include <cublas_v2.h>
#include "collision_matrices.h"

class CollisionOperator {
 public:
  virtual ~CollisionOperator() {};
  virtual void rhs(MomentsG* G, Fields* f, Geometry* geo, MomentsG* GRhs, bool accumulate) = 0;
};

class LorentzCollisionOperator : public CollisionOperator {
 public:
  LorentzCollisionOperator(Parameters* pars, Grids* grids);
  ~LorentzCollisionOperator();
  void applyCollisionMatrix(cuComplex* G_in, cuComplex* G_res, bool accumulate);
  void rhs(MomentsG* G, Fields* f, Geometry* geo, MomentsG* GRhs, bool accumulate);

 private:
  Grids* grids_;
  cuComplex* collMat{};
  cublasHandle_t handle;

  MomentsG* tmpG{};

  const int Nlm;
  const int Nk;
  dim3 dimGrid, dimBlock;
  int sharedSize;
  size_t maxSharedSize;

  void initCollisionMatrix(cuComplex* mat);
};
