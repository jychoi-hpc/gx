#include "catch.h"
#include <vector>

class Regression {
 public:
  Regression();

  // not sure if individual tests should be functions or derived classes
  void kh01();
  void other_test();
  void other_test1();
  std::vector<float> omega;
  std::vector<float> gamma;
 private:
  void execute_gx(char *);
};
