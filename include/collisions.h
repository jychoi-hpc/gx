#pragma once
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "grids.h"
#include "species.h"
#include <cublas_v2.h>
#include <cutensor.h>
#include "reductions.h"

#include <unordered_map>
#include <vector>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

class CollisionOperator {
 public:
  virtual ~CollisionOperator() {};
  virtual void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate) = 0;
};

class LorentzCollisionOperator : public CollisionOperator {
 public:
  LorentzCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob);
  ~LorentzCollisionOperator();
  void apply_collision_matrix_cublas(cuComplex* G_in, cuComplex* G_res, bool accumulate);
  void apply_collision_matrix_cutensor(cuComplex* G_in, cuComplex* G_res, bool accumulate);
  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate);

 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  cuComplex* collMat{};
  cublasHandle_t handle;

  const int Nlm;
  const int Nk;
  const specie sp_a;
  const int is_a;
  dim3 dimGrid, dimBlock;

  void init_collision_matrix(cuComplex* mat);

  // set up tensors
  // C = H_res
  // A = H_in
  // B = collMat
  // CUDA types
  cudaDataType_t typeC = CUDA_C_32F;
  cudaDataType_t typeA = CUDA_C_32F;
  cudaDataType_t typeB = CUDA_C_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;
  // vector of modes for each tensor
  std::vector<int> modeC{'x', 'l', 'm'};
  std::vector<int> modeA{'x', 'j', 'k'};
  std::vector<int> modeB{'j', 'k', 'l', 'm'};

  cutensorHandle_t cutensorHandle;
  cutensorTensorDescriptor_t descA, descB, descC;
  uint32_t alignmentRequirementA, alignmentRequirementB, alignmentRequirementC;
  cutensorContractionDescriptor_t desc;
  cutensorContractionFind_t find;
  cutensorContractionPlan_t plan;
  void *work = nullptr;
  size_t worksize = 0;

  bool first = true;
};

class AbelCollisionOperator : public CollisionOperator {
 public:
  AbelCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob);
  ~AbelCollisionOperator();
  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate);

 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  cuComplex* collMat{};
  cuComplex* collFLRMat{};
  MomentsG* energyConsG{};
  MomentsG* uparLConsG{};
  MomentsG* uparDConsG{};
  MomentsG* uperpLConsG{};
  MomentsG* uperpDConsG{};
  MomentsG* tmpG{};
  cuComplex* tmpF{};
  float* alphaE{};
  float* alphaParL{};
  float* alphaParD{};
  float* alphaPerpL{};
  float* alphaPerpD{};
  float energy_norm;
  float momentumL_norm;
  float momentumD_norm;

  const int Nlm;
  const int Nk;
  const specie sp_a;
  const int is_a;
  dim3 dimGrid, dimBlock;

  void init_collision_matrix(cuComplex* coll, cuComplex* collFLR);
  void init_conservation(MomentsG* uparLConsG, MomentsG* uparDConsG, MomentsG* uperpLConsG, MomentsG* uperpDConsG, MomentsG* energyConsG);

  void apply_collision_matrix(cuComplex* G_in, cuComplex* collMat, cuComplex* G_res, bool accumulate);
  void conservation(MomentsG* G_in, MomentsG* consG, MomentsG* G_res, float norm);

  cublasHandle_t cublasHandle;

  Reduction<cuComplex> *reduce{};

//  // set up tensors
//  // C = H_res
//  // A = H_in
//  // B = collMat
//  // CUDA types
//  cudaDataType_t typeC = CUDA_C_32F;
//  cudaDataType_t typeA = CUDA_C_32F;
//  cudaDataType_t typeB = CUDA_C_32F;
//  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;
//  // vector of modes for each tensor
//  std::vector<int> modeC{'x', 'l', 'm'};
//  std::vector<int> modeA{'x', 'j', 'k'};
//  std::vector<int> modeB{'j', 'k', 'l', 'm'};
//
//  cutensorHandle_t cutensorHandle;
//  cutensorTensorDescriptor_t descA, descB, descC;
//  uint32_t alignmentRequirementA, alignmentRequirementB, alignmentRequirementC;
//  cutensorContractionDescriptor_t desc;
//  cutensorContractionFind_t find;
//  cutensorContractionPlan_t plan;
//  void *work = nullptr;
//  size_t worksize = 0;
//
//  bool first = true;
};
