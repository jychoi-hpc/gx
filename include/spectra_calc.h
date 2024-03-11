#pragma once
#include "grids.h"
#include "ncdf.h"
#include "reductions.h"
#include "netcdf_par.h"
#include "netcdf.h"
#include "get_error.h"
#include "fields.h"
#include "moments.h"
#include <string>

using namespace std;

// base class for calculating and writing a particular kind of spectra
class SpectraCalc {
 public:
  virtual ~SpectraCalc();
  virtual void allocate();
  virtual int define_nc_variable(string varstem, int nc_group, string description = "");
  virtual void write(float *fullData, int varid, size_t time_index, int nc_group, bool isMoments, bool skip=false);
  virtual float* get_data() {return cpu;};
  virtual void dealias_and_reorder(float *fold, float *fnew) {
    for(int i=0; i<Nwrite; i++) {
      fnew[i] = fold[i];
    }
  };

 protected:
  string tag;
  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  size_t dummy_count[6] = {0};
  size_t dummy_start[6] = {0};
  std::vector<int32_t> field_species_modes{'y', 'x', 'z', 's'};
  std::vector<int32_t> field_modes{'y', 'x', 'z'};
  std::vector<int32_t> moment_species_modes{'y', 'x', 'z', 'l', 'm', 's'};
  std::vector<int32_t> moment_modes{'y', 'x', 'z', 'l', 'm'};
  std::vector<int32_t> reduced_modes;

  Grids *grids_;
  Reduction<float> *field_reduce, *moments_reduce;
 
  float *data, *tmp, *cpu;
};

// class for calculating and writing spectra of form f(s, t)
class SpectraCalc_st : public SpectraCalc {
 public:
  SpectraCalc_st(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(kx, s, t)
class SpectraCalc_kxst : public SpectraCalc {
 public:
  SpectraCalc_kxst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(ky, s, t)
class SpectraCalc_kyst : public SpectraCalc {
 public:
  SpectraCalc_kyst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(kx, ky, s, t)
class SpectraCalc_kxkyst : public SpectraCalc {
 public:
  SpectraCalc_kxkyst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(z, s, t)
class SpectraCalc_zst : public SpectraCalc {
 public:
  SpectraCalc_zst(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(kx, ky, z, s, t)
class SpectraCalc_kxkyzst : public SpectraCalc {
 public:
  SpectraCalc_kxkyzst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(l, s, t)
class SpectraCalc_lst : public SpectraCalc {
 public:
  SpectraCalc_lst(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(m, s, t)
class SpectraCalc_mst : public SpectraCalc {
 public:
  SpectraCalc_mst(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(l, m, s, t)
class SpectraCalc_lmst : public SpectraCalc {
 public:
  SpectraCalc_lmst(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(t)
class SpectraCalc_t : public SpectraCalc {
 public:
  SpectraCalc_t(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(kx, t)
class SpectraCalc_kxt : public SpectraCalc {
 public:
  SpectraCalc_kxt(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(ky, t)
class SpectraCalc_kyt : public SpectraCalc {
 public:
  SpectraCalc_kyt(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};
 
// class for calculating and writing spectra of form f(kx, ky, t)
class SpectraCalc_kxkyt : public SpectraCalc {
 public:
  SpectraCalc_kxkyt(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for calculating and writing spectra of form f(z, t)
class SpectraCalc_zt : public SpectraCalc {
 public:
  SpectraCalc_zt(Grids *grids, NcDims *nc_dims);
};

// class for calculating and writing spectra of form f(kx, ky, z, t)
class SpectraCalc_kxkyzt : public SpectraCalc {
 public:
  SpectraCalc_kxkyzt(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};


// class that packages and initialzes all spectra calculator classes
class AllSpectraCalcs {
 public:
  AllSpectraCalcs(Grids *grids, NcDims *nc_dims) {
    st_spectra = new SpectraCalc_st(grids, nc_dims);
    kxst_spectra = new SpectraCalc_kxst(grids, nc_dims);
    kyst_spectra = new SpectraCalc_kyst(grids, nc_dims);
    kxkyst_spectra = new SpectraCalc_kxkyst(grids, nc_dims);
    zst_spectra = new SpectraCalc_zst(grids, nc_dims);
    kxkyzst_spectra = new SpectraCalc_kxkyzst(grids, nc_dims);

    t_spectra = new SpectraCalc_t(grids, nc_dims);
    kxt_spectra = new SpectraCalc_kxt(grids, nc_dims);
    kyt_spectra = new SpectraCalc_kyt(grids, nc_dims);
    kxkyt_spectra = new SpectraCalc_kxkyt(grids, nc_dims);
    zt_spectra = new SpectraCalc_zt(grids, nc_dims);
    kxkyzt_spectra = new SpectraCalc_kxkyzt(grids, nc_dims);

    lst_spectra = new SpectraCalc_lst(grids, nc_dims);
    mst_spectra = new SpectraCalc_mst(grids, nc_dims);
    lmst_spectra = new SpectraCalc_lmst(grids, nc_dims);
  };
  ~AllSpectraCalcs() {
    delete st_spectra;
    delete kxst_spectra;
    delete kyst_spectra;
    delete kxkyst_spectra;
    delete zst_spectra;
    delete kxkyzst_spectra;

    delete t_spectra;
    delete kxt_spectra;
    delete kyt_spectra;
    delete kxkyt_spectra;
    delete zt_spectra;
    delete kxkyzt_spectra;

    delete lst_spectra;
    delete mst_spectra;
    delete lmst_spectra;
  };

  SpectraCalc * st_spectra;
  SpectraCalc * kxst_spectra;
  SpectraCalc * kyst_spectra;
  SpectraCalc * kxkyst_spectra;
  SpectraCalc * zst_spectra;
  SpectraCalc * kxkyzst_spectra;

  SpectraCalc * t_spectra;
  SpectraCalc * kxt_spectra;
  SpectraCalc * kyt_spectra;
  SpectraCalc * kxkyt_spectra;
  SpectraCalc * zt_spectra;
  SpectraCalc * kxkyzt_spectra;

  SpectraCalc * lst_spectra;
  SpectraCalc * mst_spectra;
  SpectraCalc * lmst_spectra;
};
