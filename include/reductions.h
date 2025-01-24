#pragma once
#include <stdio.h>
#include <stdlib.h>
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

#if (CUTENSOR_VERSION >= 20000)
template <class T> static cutensorComputeDescriptor_t computeType();
template <> cutensorComputeDescriptor_t computeType<float>()
{
  return CUTENSOR_COMPUTE_DESC_32F;
}
template <> cutensorComputeDescriptor_t computeType<double>()
{
  return CUTENSOR_COMPUTE_DESC_64F;
}
template <class T> static cutensorDataType_t dataType();
template <> cutensorDataType_t dataType<float>()
{
  return CUTENSOR_R_32F;
}
template <> cutensorDataType_t dataType<double>()
{
  return CUTENSOR_R_64F;
}
#else
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
#endif

template <class T> class Reduction {
 public:
  Reduction(Grids *grids, std::vector<int32_t> modeFull, std::vector<int32_t> modeReduced, int N=0);
  ~Reduction();
  void Sum(T* f, T* res);
  void Max(T* f, T* res);
 private:
  void * Addwork;     uint64_t sizeAdd;    uint64_t sizeWork;    
  void * Maxwork;     uint64_t sizeMax;
#if (CUTENSOR_VERSION >= 20000)
  cutensorDataType_t cfloat = dataType<T>();
  cutensorComputeDescriptor_t typeCompute = computeType<T>();
#else
  cudaDataType_t cfloat = dataType<T>();
  cutensorComputeType_t typeCompute = computeType<T>();
#endif
  cutensorOperator_t opAdd = CUTENSOR_OP_ADD;
  cutensorOperator_t opMax = CUTENSOR_OP_MAX;
#if (CUTENSOR_VERSION >= 20000)
  cutensorHandle_t handle; 
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
  cutensorHandle_t *handle; 
#else
  cutensorHandle_t handle; 
#endif

#if (CUTENSOR_VERSION >= 20000)
  cutensorPlanPreference_t options;
  cutensorPlan_t sumPlan,maxPlan;
#else
  cutensorContractionFind_t find;
#endif
    
  T alpha = 1.0;
  T beta  = 0.0;
  
  Grids *grids_;
  bool initialized_Sum = false;
  bool initialized_Max = false;

  cutensorTensorDescriptor_t descFull, descReduced;
#if (CUTENSOR_VERSION >= 20000)
  cutensorOperationDescriptor_t sumDesc,maxDesc;
#endif

  std::vector<int64_t> extentFull, extentReduced;

  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int32_t> modeFull_;
  std::vector<int32_t> modeReduced_;

  bool reduce_m, reduce_s;
  int nelementsReduced;
  int N_;
};

