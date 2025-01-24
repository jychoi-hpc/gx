#pragma once

typedef struct {
  float z; 
  float mass;
  float dens;
  float temp;
  float tprim;
  float fprim;
  float nu_ss;
  float rho2;           
  float rho2_long_wavelength_GK;
  float tz;
  float zt;
  float vt;
  float nt;
  float nz;
  float jparfac;
  float jperpfac;
  int type;
} specie;
