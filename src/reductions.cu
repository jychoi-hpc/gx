#include "reductions.h"
#include <iostream>

template<class T> Reduction<T>::Reduction(Grids *grids, std::vector<int32_t> modeFull, std::vector<int32_t> modeReduced, int N) 
 : grids_(grids), modeFull_(modeFull), modeReduced_(modeReduced), N_(N)
{
  Addwork = nullptr;     sizeWork = 0;         sizeAdd = 0;
  Maxwork = nullptr;     sizeMax = 0;

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

#if (CUTENSOR_VERSION >= 20000)
  HANDLE_ERROR(cutensorCreate(&handle));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descFull, modeFull_.size(), extentFull.data(), NULL, cfloat, 256 ));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descReduced, modeReduced_.size(), extentReduced.data(), NULL, cfloat, 256 ));
  HANDLE_ERROR(cutensorCreatePlanPreference( handle, &options, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT ) );
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
  HANDLE_ERROR(cutensorCreate(&handle));
  HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descFull, modeFull_.size(), extentFull.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
  HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descReduced, modeReduced_.size(), extentReduced.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
#else
  HANDLE_ERROR(cutensorInit(&handle));
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descFull, modeFull_.size(), extentFull.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descReduced, modeReduced_.size(), extentReduced.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
#endif
}

template<class T> Reduction<T>::~Reduction()
{
#if (CUTENSOR_VERSION >= 20000)
  HANDLE_ERROR(cutensorDestroyTensorDescriptor( descFull ));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor( descReduced ));
  HANDLE_ERROR(cutensorDestroyPlanPreference( options ));
  HANDLE_ERROR(cutensorDestroy( handle ));
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
  HANDLE_ERROR(cutensorDestroy( handle ));
#endif

  if (Addwork) cudaFree(Addwork);
  if (Maxwork) cudaFree(Maxwork);
}

template<class T> void Reduction<T>::Sum(T* dataFull, T* dataReduced)
{
  if (!initialized_Sum) {
  
#if (CUTENSOR_VERSION >= 20000)
    HANDLE_ERROR(cutensorCreateReduction( handle, &sumDesc,
                                    descFull, modeFull_.data(), CUTENSOR_OP_IDENTITY,
                                    descReduced, modeReduced_.data(), CUTENSOR_OP_IDENTITY,
                                    descReduced, modeReduced_.data(),
                                    opAdd, typeCompute ));
    HANDLE_ERROR(cutensorEstimateWorkspaceSize( handle, sumDesc, options, CUTENSOR_WORKSPACE_DEFAULT, &sizeAdd));
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
    HANDLE_ERROR(cutensorReductionGetWorkspaceSize(handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opAdd, typeCompute, &sizeAdd));
#else
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opAdd, typeCompute, &sizeAdd));
#endif
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;	sizeWork = 0;
      }
    }
#if (CUTENSOR_VERSION >= 20000)
    HANDLE_ERROR( cutensorCreatePlan( handle, &sumPlan, sumDesc, options, sizeWork ) );
#endif
    initialized_Sum = true;
  }
  
#if (CUTENSOR_VERSION >= 20000)
  HANDLE_ERROR( cutensorReduce( handle, sumPlan, &alpha, dataFull, &beta, dataReduced, dataReduced, Addwork, sizeWork, 0 ) );
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
  HANDLE_ERROR(cutensorReduction(handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0));
#else
  HANDLE_ERROR(cutensorReduction(&handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0));
#endif

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
  
#if (CUTENSOR_VERSION >= 20000)
    HANDLE_ERROR(cutensorCreateReduction( handle, &maxDesc,
                                    descFull, modeFull_.data(), CUTENSOR_OP_IDENTITY,
                                    descReduced, modeReduced_.data(), CUTENSOR_OP_IDENTITY,
                                    descReduced, modeReduced_.data(),
                                    opMax, typeCompute ));
    HANDLE_ERROR(cutensorEstimateWorkspaceSize( handle, maxDesc, options, CUTENSOR_WORKSPACE_DEFAULT, &sizeMax));
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
    HANDLE_ERROR(cutensorReductionGetWorkspaceSize(handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opMax, typeCompute, &sizeMax));
#else
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, dataFull, &descFull, modeFull_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  dataReduced, &descReduced, modeReduced_.data(),
				  opMax, typeCompute, &sizeMax));
#endif
    if (sizeMax > sizeWork) {
      sizeWork = sizeMax;
      if (Maxwork) cudaFree (Maxwork);
      if (cudaSuccess != cudaMalloc(&Maxwork, sizeWork)) {
        Maxwork = nullptr;
        sizeWork = 0;
      }
    }

#if (CUTENSOR_VERSION >= 20000)
    HANDLE_ERROR( cutensorCreatePlan( handle, &maxPlan, maxDesc, options, sizeWork ) );
#endif
    initialized_Max = true;
  }
  
#if (CUTENSOR_VERSION >= 20000)
  HANDLE_ERROR( cutensorReduce( handle, maxPlan, &alpha, dataFull, &beta, dataReduced, dataReduced, Maxwork, sizeWork, 0 ) );
#elif (defined(__HIPCC__) | CUTENSOR_VERSION >= 10700)
  HANDLE_ERROR(cutensorReduction(handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opMax, typeCompute, Maxwork, sizeMax, 0));
#else
  HANDLE_ERROR(cutensorReduction(&handle,
		    (const void*) &alpha, dataFull, &descFull, modeFull_.data(),
		    (const void*) &beta,  dataReduced, &descReduced, modeReduced_.data(),
		    dataReduced,  &descReduced, modeReduced_.data(),
		    opMax, typeCompute, Maxwork, sizeMax, 0));
#endif

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

template class Reduction<float>;
template class Reduction<double>;

