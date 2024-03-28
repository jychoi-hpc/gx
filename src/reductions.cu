#include "reductions.h"
#include <iostream>

template<class T> Reduction<T>::Reduction(Grids *grids, std::vector<int32_t> modeFull, std::vector<int32_t> modeReduced, int N) 
 : grids_(grids), modeFull_(modeFull), modeReduced_(modeReduced), N_(N)
{
  Addwork = nullptr;     sizeWork = 0;         sizeAdd = 0;
  Maxwork = nullptr;     sizeMaxWork = 0;      sizeMax = 0;

  // initialize all possible extents
  extent['y'] = grids_->Nyc;
  extent['r'] = grids_->Ny;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  extent['l'] = grids_->Nl;
  extent['m'] = grids_->Nm;
  extent['s'] = grids_->Nspecies;
  extent['a'] = N_;

  // whether a reduction over m index is required (which may be parallelized)
  reduce_m = false;
  // whether a reduction over s index is required (which may be parallelized)
  reduce_s = false;

  // create a vector of extents for the full tensor
  for (auto mode : modeFull_) {
    extentFull.push_back(extent[mode]);;
    if(mode == 'm') {
      reduce_m = true;
    }
    if(mode == 's') {
      reduce_s = true;
    }
  }

  // create a vector of extents for the reduced tensor
  nelementsReduced = 1;
  for (auto mode : modeReduced_) {
    extentReduced.push_back(extent[mode]);;
    nelementsReduced *= extent[mode];
    if(mode == 'm') {
      reduce_m = false;
    }
    if(mode == 's') {
      reduce_s = false;
    }
  }

  HANDLE_ERROR(cutensorInit(&handle));
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descFull, modeFull_.size(), extentFull.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descReduced, modeReduced_.size(), extentReduced.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
}

template<class T> Reduction<T>::~Reduction()
{
  if (Addwork) cudaFree(Addwork);
  if (Maxwork) cudaFree(Maxwork);
}

template<class T> void Reduction<T>::Sum(T* dataFull, T* dataReduced)
{
  if (!initialized_Sum) {
  
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opAdd, typeCompute, &sizeAdd));
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;	sizeWork = 0;
      }
    }
    initialized_Sum = true;
  }
  
  HANDLE_ERROR(cutensorReduction(&handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0));

  if(reduce_m && reduce_s && grids_->nprocs > 1) {
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclSum, grids_->ncclComm, 0);
  }
  // reduce across parallelized m blocks
  if(reduce_m && grids_->nprocs_m > 1 && grids_->nprocs > 1) {
    // ncclComm_s is the per-species communicator
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclSum, grids_->ncclComm_s, 0);
  }
  // reduce across parallelized s blocks
  if(reduce_s && grids_->nprocs_s > 1 && grids_->nprocs > 1) {
    // ncclComm_m is the per-m-block communicator
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclSum, grids_->ncclComm_m, 0);
  }

}		     

template<class T> void Reduction<T>::Max(T* dataFull, T* dataReduced)
{
  if (!initialized_Max) {
  
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opMax, typeCompute, &sizeMax));
    if (sizeMax > sizeMaxWork) {
      sizeMaxWork = sizeMax;
      if (Maxwork) cudaFree (Maxwork);
      if (cudaSuccess != cudaMalloc(&Maxwork, sizeMaxWork)) {
        Maxwork = nullptr;	sizeMaxWork = 0;
      }
    }
    initialized_Max = true;
  }
  
  HANDLE_ERROR(cutensorReduction(&handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opMax, typeCompute, Maxwork, sizeMax, 0));

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  if(reduce_m && reduce_s && grids_->nprocs > 1) {
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclMax, grids_->ncclComm, 0);
  }
  // reduce across parallelized m blocks
  if(reduce_m && grids_->nprocs_m > 1) {
    // ncclComm_s is the per-species communicator
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclMax, grids_->ncclComm_s, 0);
  }
  // reduce across parallelized s blocks
  if(reduce_s && grids_->nprocs_s > 1) {
    // ncclComm_m is the per-m-block communicator
    ncclAllReduce((void*) dataReduced, (void*) dataReduced, nelementsReduced, ncclFloat, ncclMax, grids_->ncclComm_m, 0);
  }
}

//============ DenseM ==============
DenseM::DenseM(int N, int M) : N_(N), M_(M)
{ 
  Multwork = nullptr;      sizeWork = 0;

  extent['g'] = M_;
  extent['r'] = N_;
  extent['s'] = N_;
  for (auto mode : Mmode) extent_M.push_back(extent[mode]); 
  for (auto mode : Vmode) extent_V.push_back(extent[mode]); 
  for (auto mode : Ymode) extent_Y.push_back(extent[mode]); 
  for (auto mode : Zmode) extent_Z.push_back(extent[mode]); 
  for (auto mode : Wmode) extent_W.push_back(extent[mode]); 
  for (auto mode : Xmode) extent_X.push_back(extent[mode]); 

  cutensorInit(&handle);
  cutensorInitTensorDescriptor(&handle, &dY, nYmode, extent_Y.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dM, nMmode, extent_M.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dV, nVmode, extent_V.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dX, nXmode, extent_X.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dW, nWmode, extent_W.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dZ, nZmode, extent_Z.data(), NULL, dfloat, CUTENSOR_OP_IDENTITY);

  cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT);
}

DenseM::~DenseM()
{
  if (Multwork) cudaFree(Multwork);  
}

void DenseM::MatMat(double* Res, double* M1, double* M2)
{
  if (first_MM) {
    
    uint32_t alignM1, alignM2, alignRes;
    cutensorGetAlignmentRequirement(&handle, M1,  &dW, &alignM1);
    cutensorGetAlignmentRequirement(&handle, M2,  &dZ, &alignM2);
    cutensorGetAlignmentRequirement(&handle, Res, &dX, &alignRes);

    cutensorInitContractionDescriptor (&handle, &dMM, 
				       &dW, Wmode.data(), alignM1,
				       &dZ, Zmode.data(), alignM2,
				       &dX, Xmode.data(), alignRes,
				       &dX, Xmode.data(), alignRes,
				       typeCompute64);

    cutensorContractionGetWorkspace(&handle, &dMM, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &sizeMM );
    if (sizeMM > 0) {
      if (cudaSuccess != cudaMalloc(&MMwork, sizeMM)) {MMwork = nullptr; sizeMM = 0;}
    }
    first_MM = false;

    cutensorInitContractionPlan(&handle, &MMplan, &dMM, &find, sizeMM);
  }
  
  cutensorContraction(&handle,
		      &MMplan, (void*) &alpha, M1, M2, (void*) &beta, Res, Res, MMwork, sizeMM, 0);
  
}

// Res[M] = Mat[M x N] * Vec[N]
// and in terms of these tensor descriptors Y = M V
void DenseM::MatVec(double* Res, double* Mat, double* Vec)
{
  if (first_MV) {
    
    uint32_t alignVec, alignMat, alignRes;
    cutensorGetAlignmentRequirement(&handle, Mat, &dM, &alignMat);
    cutensorGetAlignmentRequirement(&handle, Vec, &dV, &alignVec);
    cutensorGetAlignmentRequirement(&handle, Res, &dY, &alignRes);

    cutensorInitContractionDescriptor (&handle, &dMV, 
				       &dM, Mmode.data(), alignMat,
				       &dV, Vmode.data(), alignVec,
				       &dY, Ymode.data(), alignRes,
				       &dY, Ymode.data(), alignRes,
				       typeCompute64);

    cutensorContractionGetWorkspace(&handle, &dMV, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &sizeWork);
    if (sizeWork > 0) {
      if (cudaSuccess != cudaMalloc(&Multwork, sizeWork)) {Multwork = nullptr; sizeWork = 0;}
    }
    first_MV = false;

    cutensorInitContractionPlan(&handle, &MVplan, &dMV, &find, sizeWork);
  }
  
  cutensorContraction(&handle, &MVplan, (void*) &alpha, Mat, Vec,
		      (void*) &beta, Res, Res, Multwork, sizeWork, 0);
  
}

template class Reduction<float>;
template class Reduction<double>;

