#define CUB_STDERR

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "reductions.h"
//#include "cub/cub.cuh"

void Red::Sum(float* rmom, float* val)
{
  
  using namespace cub;
  int arr_len = 10; // hard coded array length
  d_in = NULL; // d_in and d_out are float pointers
  d_out = NULL;
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  CachingDeviceAllocator  g_allocator(true);

  // just printing out the array to be summed
  std::cout << "New Array\n";
  std::cout << "---------\n";
  for (int i=0; i<10; i++) {
    std::cout << rmom[i] << " ";
  }
  std::cout << "\n";
  
  g_allocator.DeviceAllocate((void**)&d_in, arr_len*sizeof(float));
  cudaMemcpy(d_in, rmom, arr_len*sizeof(float), cudaMemcpyHostToDevice);
  g_allocator.DeviceAllocate((void**)&d_out, 1*sizeof(float));

  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, Ng_));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Actual sum
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, Ng_));

  cudaMemcpy(val, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Sum = " << val[0] << "\n\n";

  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
}
