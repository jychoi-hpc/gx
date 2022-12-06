# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = googletest/googletest
GX_DIR = ..
GX_LIB = ${GX_DIR}/libgx.a

#######################################
# Include system-dependent make variables
#######################################

ifndef GK_SYSTEM
	ifdef SYSTEM
$(warning SYSTEM environment variable is obsolete)
$(warning use GK_SYSTEM instead)
	GK_SYSTEM = $(SYSTEM)
	else
$(error GK_SYSTEM is not set)
	endif
endif
include $(GX_DIR)/Makefiles/Makefile.$(GK_SYSTEM)

############################
## Setup Compiler Flags
###########################

CXX = g++
CC = $(NVCC)
LD = $(NVCC)

# Flags passed to nvcc compiler.
CFLAGS= ${CUDA_INC} ${MPI_INC} ${NETCDF_INC} ${CUTENSOR_INC}
LDFLAGS= $(CUDA_LIB) ${MPI_LIB} ${NETCDF_LIB} ${GSL_LIB} ${CUTENSOR_LIB}

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
GTEST_FLAGS += -isystem $(GTEST_DIR)/include

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
UNIT_TESTS = test_grids test_geometry test_grad_perp # test_trinity_interface test_grad_parallel_periodic test_grad_parallel_linked test_moments test_laguerre_transform test_grad_perp
REGRESSION_TESTS = #test_cyclone_linear_beer
TESTS = $(UNIT_TESTS) $(REGRESSION_TESTS)

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GX_HEADERS = $(GX_DIR)/include/*.h

GX_OBJS = device_funcs.o parameters.o grids.o reductions.o reservoir.o grad_perp.o fields.o moments.o forcing.o grad_parallel.o grad_parallel_linked.o geometry.o hermite_transform.o laguerre_transform.o nca.o ncdf.o solver.o smith_par_closure.o closures.o linear.o nonlinear.o ts_sspx2.o ts_sspx3.o ts_rk2.o ts_rk4.o ts_k10.o ts_k2.o ts_g3.o diagnostics.o run_gx.o version.o trinity_interface.o
GX_OBJS := $(addprefix $(GX_DIR)/obj/, $(GX_OBJS))

VPATH=.:src

# House-keeping build targets.

all : unit_tests

clean :
	rm -f test_* obj/*.a obj/*.o

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
obj/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	            $(GTEST_DIR)/src/gtest-all.cc -o $@

obj/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	            $(GTEST_DIR)/src/gtest_main.cc -o $@

obj/gtest.a : obj/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

obj/gtest_main.a : obj/gtest-all.o obj/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GX_LIB): $(GX_DIR)/src/*.cu
	cd $(GX_DIR); \
	make libgx.a; \
	cd unit_tests;

# Build tests
obj/test_%.o: test_%.cu 
	$(NVCC) -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) -I $(GX_DIR)/include $(GTEST_FLAGS)

obj/device_test.o: device_test.cu
	$(NVCC) -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) $(GTEST_FLAGS)

test_%: obj/test_%.o obj/gtest_main.a $(GX_LIB) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

unit_tests: $(addprefix obj/, $(UNIT_TESTS:=.o)) obj/gtest_main.a $(GX_LIB) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

regression_tests: $(addprefix obj/, $(REGRESSION_TESTS:=.o)) obj/gtest_main.a $(GX_LIB) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

test_each: $(TESTS)

run_tests: unit_tests
	./unit_tests;
