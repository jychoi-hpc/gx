#include "catch.h"
#include <vector>

class Regression {
 public:
  Regression();
  ~Regression();
  // not sure if individual tests should be functions or derived classes
  void kh01();
  void slab();

  // output variables
  std::vector<float> omega_;
  std::vector<float> gamma_;
  int nstep_;
  int nwrite_;
  int ny_;
  int nx_;
  int ntime_;
  int Naky_;
  int Nakx_;
  
 private:
  void execute_gx(char *);
  void read_netcdf_output(char *);

};
