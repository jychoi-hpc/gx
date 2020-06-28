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

//namespace py = pybind11;
//using namespace py::literals;

Regression::Regression() {}

void Regression::kh01() {
  // remove any previous .nc outputs
  if ("./tests/regression/kh01.nc") {
    std::remove("./tests/regression/kh01.nc");
  }
  if ("./tests/regression/kh01a.nc") {
    std::remove("./tests/regression/kh01a.nc");
  }
  
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

  read_netcdf_output("./tests/regression/kh01a.nc");

}

void Regression::slab() {
  // remove any previous .nc outputs
  if ("./tests/regression/slab.nc") {
    std::remove("./tests/regression/slab.nc");
  }

  // need to search for kh01.in and kh01a.in
  std::ifstream slab{ "./tests/regression/slab.in" };
  
  if (!slab) {
    std::cerr << "One of the required input files could not be found. Exiting...\n";
    exit(1);
  }
  // potentially read in relevant input variables (k-values) to ensure that these are not somehow altered when running the code

  // call function to execute GX for input file
  execute_gx("./tests/regression/slab");
  
  // read in the output file
  if (!"./tests/regression/slab.nc") {
    std::cerr << "The output file slab.nc was not found. Exiting...";
    exit(1);
  }

  read_netcdf_output("./tests/regression/slab.nc");

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

void Regression::read_netcdf_output(char *out_file) {
  int nc_id;
  int var_id;

  std::cout << out_file << "\n";
  nc_open(out_file, NC_NOWRITE, &nc_id);
  // get information about nstep/nwrite s
  nc_inq_varid(nc_id, "nstep", &var_id);
  nc_get_var_int(nc_id, var_id, &nstep_);
  std::cout << "nstep = " << nstep_ << ", ";
  
  nc_inq_varid(nc_id, "nwrite", &var_id);
  nc_get_var_int(nc_id, var_id, &nwrite_);
  std::cout << "nwrite = " << nwrite_ << ", ";
  
  nc_inq_varid(nc_id, "ny", &var_id);
  nc_get_var_int(nc_id, var_id, &ny_);
  std::cout << "ny = " << ny_ << ", ";
  
  nc_inq_varid(nc_id, "nx", &var_id);
  nc_get_var_int(nc_id, var_id, &nx_);
  std::cout << "nx = " << nx_ << ", ";
  
  ntime_ = (nstep_/nwrite_);
  std::cout << "ntime = " << ntime_ << "\n";
  float time_data_[ntime_];

  nc_inq_varid(nc_id, "time", &var_id);
  nc_get_var_float(nc_id, var_id, &time_data_[0]);
  
  std::cout << "Time: ";
  for (int i=0; i<ntime_; i++) {
    std::cout << time_data_[i] << ", ";
  }
  std::cout << "\n";
  
  // Number of simulated modes
  Naky_ = (ny_-1)/3 + 1;
  Nakx_ = (2*(nx_-1)/3) + 1;
  std::cout << "Naky = " << Naky_ << ", ";
  std::cout << "Nakx = " << Nakx_ << "\n";

  float omega_v_time_[Naky_*Nakx_*(ntime_+1)][2];
  nc_inq_varid(nc_id, "omega_v_time", &var_id);
  nc_get_var_float(nc_id, var_id, &omega_v_time_[0][0]);

  for (int z=0; z<Naky_*Nakx_; z++) {
    omega_.push_back(omega_v_time_[Nakx_*Naky_*(ntime_)+z][0]);
    gamma_.push_back(omega_v_time_[Nakx_*Naky_*(ntime_)+z][1]);
  }

  nc_close(nc_id);  
}

Regression::~Regression() {}
