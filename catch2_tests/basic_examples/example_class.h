#include <fstream>
#include "catch.hpp"
#include <vector>

class Grids {
 public:
  Grids();
  int simple_function(int);
  std::ifstream in_file;
  void get_filename(std::vector<int>&);
  int get_netcdf_varname();
  int something;
  int other{};
};
  
