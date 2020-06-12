#include <stdio.h>
#include <limits>
#include <typeinfo>
#include "grids.h"

class Red {
 public:
 Red(Grids* grids) : grids_(grids), Ng_(grids_->NxNycNz) {};
  void Sum(float* rmom, float *val);

 private:
  float* d_in;
  float* d_out;
  void* d_temp_storage;
  size_t temp_storage_bytes;
  const Grids* grids_;
  const size_t Ng_;
};  
