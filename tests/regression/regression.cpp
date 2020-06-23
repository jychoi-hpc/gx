#include "catch.h"
#include "regression.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <netcdf.h>
#include <string>
#include <tuple>

// includes for fork() to execute GX
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/wait.h>

namespace py = pybind11;
using namespace py::literals;

//void execute_gx(char *);

Regression::Regression() {};

void Regression::kh01() {
  // need to search for kh01.in and kh01a.in
  std::ifstream kh01{ "./tests/regression/kh01.in" };
  std::ifstream kh01a{ "./tests/regression/kh01a.in" };
  
  if (!kh01 or !kh01a) {
    std::cerr << "One of the required input files could not be found. Exiting...\n";
    exit(1);
  }
  // potentially read in relevant input variables (k-values) to ensure that these are not somehow altered when running the code

  // call function to execute GX for input file
  execute_gx("./tests/regression/kh01");
  execute_gx("./tests/regression/kh01a");

  /*
  // call python function to remove previous runs and execute tests
  int test_variable;
  int x;
  auto python_vars = py::dict(); // test line to check that embedded python works
  // the following section imports the python script "run_regression_tests.py" and then calls the function kh01 of that script, which runs the GX test cases
  py::exec(R"(
import python_test as py_module
from scipy.io import netcdf
py_module.kh01()
tuples = py_module.return_tuple()
test_var = 4
)", py::globals(), python_vars);
  test_variable = python_vars["test_var"].cast<int>();
  std::cout << test_variable << "\n";
  */

  // read in the output file
  if (!"./tests/regression/kh01a.nc") {
    std::cerr << "The output file kh01a.nc was not found. Exiting...";
    exit(1);
  }
  int nc_id{};
  int var_id{};
  int nstep, nwrite, ny, nx, ntime;
  nc_open("./tests/regression/kh01a.nc", NC_NOWRITE, &nc_id);
  // get information about nstep/nwrite s
  nc_inq_varid(nc_id, "nstep", &var_id);
  nc_get_var_int(nc_id, var_id, &nstep);
  nc_inq_varid(nc_id, "nwrite", &var_id);
  nc_get_var_int(nc_id, var_id, &nwrite);
  nc_inq_varid(nc_id, "ny", &var_id);
  nc_get_var_int(nc_id, var_id, &ny);
  nc_inq_varid(nc_id, "nx", &var_id);
  nc_get_var_int(nc_id, var_id, &nx);
  ntime = (nstep/nwrite);
  float time_data[ntime];
  nc_inq_varid(nc_id, "time", &var_id);
  nc_get_var_float(nc_id, var_id, &time_data[0]);
  /*  std::cout << "Time: ";
  for (int i=0; i<ntime; i++) {
    std::cout << time_data[i] << ", ";
  }
  std::cout << "\n";*/
  // Number of simulated modes
  int Naky = (ny-1)/3 + 1;
  int Nakx = (2*(nx-1)/3) + 1;
  float omega_v_time[Naky*Nakx*ntime][2];
  nc_inq_varid(nc_id, "omega_v_time", &var_id);
  nc_get_var_float(nc_id, var_id, &omega_v_time[0][0]);
  for (int z=0; z<Naky*Nakx; z++) {
    omega.push_back(omega_v_time[Nakx*Naky*ntime+z][0]);
    gamma.push_back(omega_v_time[Nakx*Naky*ntime+z][1]);
  }
  nc_close(nc_id);
}

void Regression::execute_gx(char *in_file) {
  pid_t process_id;
  int return_val = 1;
  int state;

  process_id = fork();
  if (process_id == -1) { //when process id is negative, there is an error, unable to fork
    printf("can't fork, error occured\n");
    exit(EXIT_FAILURE);
    
  } else if (process_id == 0) { //the child process is created
    char * argv_list[] = {"./tests/unit_tests/gx",in_file, NULL};
    execv("./tests/unit_tests/gx",argv_list);
    exit(0);
    
  } else { //for the parent process
    if (waitpid(process_id, &state, 0) > 0) { //wait untill the process change its state
      if (WIFEXITED(state) && !WEXITSTATUS(state))
	printf("program is executed successfully\n");
      else if (WIFEXITED(state) && WEXITSTATUS(state)) {
	if (WEXITSTATUS(state) == 127) {
	  printf("Execution failed\n");
	} else
	  printf("program terminated with non-zero status\n");
      } else
	printf("program didn't terminate normally\n");
    }
    else {
      printf("waitpid() function failed\n");
    }
  }
}
