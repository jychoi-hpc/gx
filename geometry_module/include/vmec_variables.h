#pragma once

class VMEC_variables {
 public:
  VMEC_variables(char*);
  ~VMEC_variables();
  void sanity_check(double*, double*, int*, int*, int*, int*, double*);
  char *vmec_file;
  
  // variables to be read from VMEC file
  int lasym;
  int ns;
  int nfp;
  int mnmax;
  int mnmax_nyq;
  int signgs;
  int mpol;
  int ntor;
  double Aminor_p;
  double* iotas;
  double* iotaf;
  double* presf;
  int* xm_nyq;
  int* xn_nyq;
  int* xm;
  int* xn;
  double* phips;
  double* phi;
  double* lmns;
  double* bmnc;
  double* rmnc;
  double* zmns;
  double* gmnc;
  double* bsupumnc;
  double* bsupvmnc;
  double* bsubumnc;
  double* bsubvmnc;
  double* bsubsmns;
  
  // non-stellarator-symmetric terms
  double* lmnc;
  double* bmns;
  double* rmns;
  double* zmnc;
  double* gmns;
  double* bsupumns;
  double* bsupvmns;
  double* bsubumns;
  double* bsubvmns;
  double* bsubsmnc;
};
