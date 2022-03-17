#pragma once
#include <iostream>
#include "mpi.h"
#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "diagnostics.h"
#include "fields.h"
#include "moments.h"
#include "solver.h"
#include "forcing.h"
#include "timestepper.h"
#include "linear.h"
#include "nonlinear.h"
#include "reservoir.h"
#include "device_funcs.h"
#include "get_error.h"

void run_gx(Parameters * parameters, Grids * grids, Geometry * geo, Diagnostics * diagnostics);
