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
  if( err != CUTENSOR_STATUS_SUCCESS )                                                 \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

/*
struct CustomMax
{
  template <typename T>
  __device__ __forceinline__
  T operator()(const T &a, const T &b) const {
    return (abs(b) > abs(a)) ? abs(b) : abs(a);
  }
};
*/

class Red {
public:
  Red(Grids *grids, std::vector<int> s, bool potential);
  Red(Grids *grids, std::vector<int> s, float dummy);
  Red(Grids *grids, std::vector<int> s);
  Red(int N, int ns = 1);
  ~Red();
  void  Sum(float* rmom, float* val, int ispec); // WSPECTRA
  void pSum(float* rmom, float* val, int ispec); // PSPECTRA
  void iSum(float* rmom, float* val, int ispec); // ASPECTRA
  void sSum(float* Q, float* R); // d^3 r, species by species
  void aSum(float* A2, float* val); // sum over all elements
  void  Max(float* A2, float* val); // CFL
  
private:

  int N_;
  Grids *grids_;
  std::vector<int> spectra_;
  
  cudaDataType_t cfloat = CUDA_R_32F;;
  cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;
  float alpha = 1.;
  float beta  = 0.;

  // incoming types
  std::vector<int32_t> Wmode{'y', 'x', 'z', 'l', 'm', 's'};
  std::vector<int32_t> Pmode{'y', 'x', 'z', 's'};
  std::vector<int32_t> Imode{'y', 'x', 'z'};
  std::vector<int32_t> Amode{'a'};
  std::vector<int32_t> Bmode{};
  std::vector<int32_t> Qmode{'a', 's'};
  std::vector<int32_t> Rmode{'s'};

  std::vector<int32_t> initialized;
  bool first_Max = true;
  bool first_Sum = true;
  bool first_SumT = true;
  
  //
  // These control arrays are ordered according to the WSPECTRA, PSPECTRA, and ASPECTRA definitions in parameters.h
  // 
  std::vector<std::vector<int32_t>> Modes{{'s'},
					  {'x', 's'},
					  {'y', 's'},
					  {'z', 's'},
					  {'l', 's'},
					  {'m', 's'},
					  {'l', 'm', 's'},
					  {'y', 'x', 's'},
					  {'y', 'x', 's'},
					  {'z', 's'},
					  {'z', 's'}};
  std::vector<std::vector<int32_t>> pModes{{'s'},
					   {'x', 's'},
					   {'y', 's'},
					   {'y', 'x', 's'},
					   {'y', 'x', 's'},
					   {'z', 's'},
					   {'z', 's'}};							       
  std::vector<std::vector<int32_t>> iModes{{},
					   {'x'},
					   {'y'},
					   {'y', 'x'},
					   {'y', 'x'},
					   {'z'},
					   {'z'}};
							       
    
  int32_t nWmode = Wmode.size(); // for integrals of g**2, all species, or all (m,s), or all (l,m,s), etc.
  int32_t nPmode = Pmode.size(); // for integrals of (1-Gamma_0) Phi**2
  int32_t nImode = Imode.size(); // for integrals of Phi**2
  int32_t nAmode = Amode.size(); // for single-block data input (such as for CFL condition)
  int32_t nBmode = Bmode.size(); // scalar output for contiguous data
  int32_t nQmode = Qmode.size(); // for single-block data input (such as flux)
  int32_t nRmode = Rmode.size(); // for single-block data input (such as flux)
  
  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_W, extent_P, extent_A, extent_B, extent_Q, extent_R, extent_I;
  std::vector<std::vector<int64_t>> extents;
  char version_red;
  static cutensorHandle_t handle;
  static bool isCuTensorInitialised;
  
  std::vector<cutensorTensorDescriptor_t> desc; // for WSPECTRA, PSPECTRA, ASPECTRA
  cutensorTensorDescriptor_t dW; // for data like G**2 
  cutensorTensorDescriptor_t dP;
  cutensorTensorDescriptor_t dI;
  cutensorTensorDescriptor_t dA;
  cutensorTensorDescriptor_t dB;
  cutensorTensorDescriptor_t dQ;
  cutensorTensorDescriptor_t dR;

  const cutensorOperator_t opAdd = CUTENSOR_OP_ADD;
  const cutensorOperator_t opMax = CUTENSOR_OP_MAX;

  std::vector<uint64_t> sAdd;
  uint64_t sizeWork = 0;  
  uint64_t sizeAddT = 0;  void * AddworkT = nullptr;
  uint64_t sizeAdd = 0;   void * Addwork  = nullptr;
  uint64_t sizeMax = 0;   void * Maxwork  = nullptr;
};  
