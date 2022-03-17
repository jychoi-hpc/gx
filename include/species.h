#pragma once
#include "precision_types.h"

typedef struct {
  float z; 
  float mass;
  float dens;
  float temp;
  float tprim;
  float fprim;
  float uprim;
  float nu_ss;
  float rho2;           
  float tz;
  float zt;
  float vt;
  float nt;
  float nz;
  float qneut;
  float as;
  float amp;
  int type;
} specie;
