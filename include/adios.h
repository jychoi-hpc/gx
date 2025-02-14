#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include <string>
#include <adios2.h>

class Adios {
 public:
  Adios(Parameters* pars, Grids* grids, std::string prefix, int dims, Geometry *geo = nullptr, std::string engine = "BPFile");
  ~Adios();

  adios2::Dims shape;
  adios2::Dims start;
  adios2::Dims count;

  adios2::IO io;
  adios2::Engine writer;

  adios2::Variable<float> phiVar;
  adios2::Variable<float> aparVar;
  adios2::Variable<float> bparVar;
  adios2::Variable<double> timeVar;
  std::vector<adios2::Variable<float>> spectraVars;

  void write_time(double);
  void SetShapeFieldsKREHM();
  int AddSpectraGKVar(std::string name, int ndims, size_t sh[], size_t c[], size_t st[]);

 private:
  adios2::ADIOS adios_;
  std::string engine_;

  Parameters *pars_;
  Grids *grids_;
  Geometry *geo_;

  void defineGeometryVar();
  void defineGridVar();
};

