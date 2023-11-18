#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <unordered_map>
#include <vector>
#include <iterator>
#include "grids.h"

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

template <class T> static cutensorComputeType_t computeType();
template <> cutensorComputeType_t computeType<float>()
{
  return CUTENSOR_COMPUTE_32F;
}

template <> cutensorComputeType_t computeType<double>()
{
  return CUTENSOR_COMPUTE_64F;
}

template <class T> static cudaDataType_t dataType();
template <> cudaDataType_t dataType<float>()
{
  return CUDA_R_32F;
}

template <> cudaDataType_t dataType<double>()
{
  return CUDA_R_64F;
}

template <class T> class Reduction {
 public:
  Reduction(Grids *grids, std::vector<int32_t> modeFull, std::vector<int32_t> modeReduced, int N=0);
  ~Reduction();
  void Sum(T* f, T* res);
  void Max(T* f, T* res);
 private:
  void * Addwork;     uint64_t sizeAdd;    uint64_t sizeWork;    
  void * Maxwork;     uint64_t sizeMax;    uint64_t sizeMaxWork;
  cudaDataType_t cfloat = dataType<T>();
  cutensorComputeType_t typeCompute = computeType<T>();
  cutensorOperator_t opAdd = CUTENSOR_OP_ADD;
  cutensorOperator_t opMax = CUTENSOR_OP_MAX;
  cutensorHandle_t handle; 
  cutensorContractionFind_t find;
    
  T alpha = 1.0;
  T beta  = 0.0;
  
  Grids *grids_;
  bool initialized_Sum = false;
  bool initialized_Max = false;

  cutensorTensorDescriptor_t descFull, descReduced;

  std::vector<int64_t> extentFull, extentReduced;

  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int32_t> modeFull_;
  std::vector<int32_t> modeReduced_;

  bool reduce_m, reduce_s;
  int nelementsReduced;
  int N_;
};

class DenseM {
 public:
  DenseM(int N, int M);
  ~DenseM();
  void MatVec(double* res, double* Mat, double* vec);
  void MatMat(double* res, double* M1, double* M2);

 private:
  int M_;  int N_;
  void * Multwork;    uint64_t sizeWork; 
  void * MMwork;      uint64_t sizeMM;

  bool first_MV = true;
  bool first_MM = true;
  
  cutensorTensorDescriptor_t dM, dV, dW, dX, dY, dZ;
  cutensorContractionDescriptor_t dMV, dMM;
  cutensorContractionPlan_t MVplan, MMplan;
  cutensorHandle_t handle; 
  cutensorContractionFind_t find;
  cudaDataType_t dfloat = CUDA_R_64F;
  cutensorComputeType_t typeCompute64 = CUTENSOR_COMPUTE_64F;
  double alpha = 1.0;
  double beta  = 0.0;

  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_Y, extent_M, extent_V, extent_W, extent_X, extent_Z;

  // G = P R2 or Y[M] = M[M x N] * V[N]
  std::vector<int32_t> Ymode{'g'};
  std::vector<int32_t> Mmode{'g', 'r'};
  std::vector<int32_t> Vmode{'r'};

  // X[M x N] = W [M x N] Z [ N x N ]  
  std::vector<int32_t> Xmode{'g', 's'};
  std::vector<int32_t> Wmode{'g', 'r'};
  std::vector<int32_t> Zmode{'r', 's'};
  
  int32_t nYmode = Ymode.size(); 
  int32_t nMmode = Mmode.size(); 
  int32_t nVmode = Vmode.size(); 
  int32_t nWmode = Wmode.size(); 
  int32_t nXmode = Xmode.size(); 
  int32_t nZmode = Zmode.size(); 
};
