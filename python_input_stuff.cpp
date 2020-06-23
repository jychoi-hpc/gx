//

auto python_vars = py::dict();
  py::exec(R"(
import sys
sys.path.insert(1, './python')
import set_defaults
import imp
input_file = imp.load_source('input_file', './input_file')
from scipy.io import netcdf
default_dict = set_defaults.set_GX_defaults("./tests/regression/kh01.nc")
print("Default Parameters    --> ",default_dict)
input_parameters = input_file.input_file(default_dict)
print("Parameters from Input --> ",input_parameters)
type_mismatch = []
type_mismatch = set_defaults.type_check(input_parameters)
if len(type_mismatch) == 0:
    print("hello")
)", py::globals(), python_vars);
  py::dict input_pars = python_vars["input_parameters"];
  int nx_val = input_pars["nx"].cast<int>();
  int ny_val = input_pars["ny"].cast<int>();
  int y0_val = input_pars["y0"].cast<float>();
  std::cout << "nx in cpp = " << nx_val << "\n";
  std::cout << "ny in cpp = " << ny_val << "\n";
  std::cout << "y0 in cpp = " << y0_val << "\n";
