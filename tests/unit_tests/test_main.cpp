#define CATCH_CONFIG_RUNNER
#include "catch.h"
//#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

int main( int argc, char* argv[] ) {
  //MPI_Init(&argc, &argv);
  py::scoped_interpreter guard{}; // cannot call python functions once this goes out of scope
  int result = Catch::Session().run( argc, argv );
  //MPI_Finalize();
  return result;
}

