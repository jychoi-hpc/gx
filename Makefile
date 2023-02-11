#######################################
#        Makefile for GX
#######################################

TARGET    = all

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
include Makefiles/Makefile.$(GK_SYSTEM)

############################
## Setup Compiler Flags
###########################

INCS= ${CUDA_INC} ${MPI_INC} ${NETCDF_INC} ${GSL_INC}
LIBS= $(CUDA_LIB) ${MPI_LIB} ${NETCDF_LIB} ${GSL_LIB} ${C_LIB}

#####################################
# Rule for building the system_config
# used by the build script
#####################################

ifdef STANDARD_SYSTEM_CONFIGURATION
system_config: Makefiles/Makefile.$(GK_SYSTEM) Makefile
	@echo "#!/bin/bash " > system_config
	@echo "$(STANDARD_SYSTEM_CONFIGURATION)" >> system_config
	@sed -i 's/^ //' system_config
else
.PHONY: system_config
system_config:
	$(error "STANDARD_SYSTEM_CONFIGURATION is not defined for this system")
endif

VPATH=.:src:geometry_modules/vmec/src

##########################
## Suffix Build Rules
#############################

.SUFFIXES:
.SUFFIXES: .c .cpp .cu .o .d
.DEFAULT_GOAL := $(TARGET)

HEADERS=$(wildcard include/*.h) 
VMEC_GEO_HEADERS = $(wildcard geometry_modules/vmec/include*.h)

ifdef GS2_PATH
obj/%.o: %.cu $(HEADERS) 
	$(NVCC) -w -dc -o $@ $< $(NVCCFLAGS) $(INCS) -I. -I include -I geometry_modules/vmec/include -DGX_PATH=\"${PWD}\" -DGS2_PATH=\"${GS2_PATH}\"
else                                        
obj/%.o: %.cu $(HEADERS)                    
	$(NVCC) -w -dc -o $@ $< $(NVCCFLAGS) $(INCS) -I. -I include -I geometry_modules/vmec/include -DGX_PATH=\"${PWD}\"
endif

obj/%.o: %.cpp $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCS) -I. -I include -I geometry_modules/vmec/include 

.SILENT: src/version.c obj/version.o

obj/version.o: src/version.c
	$(CC) -c -o $@ $< $(CFLAGS) $(INCS) -I. -I include

src/version.c: 
	git describe --always --dirty --tags | awk ' BEGIN {print "#include \"version.h\""} {print "const char * build_git_sha = \"" $$0"\";"} END {}' > src/version.c
	date | awk 'BEGIN {} {print "const char * build_git_time = \""$$0"\";"} END {} ' >> src/version.c
	whoami | awk 'BEGIN {} {print "const char * build_user = \""$$0"\";"} END {} ' >> src/version.c
	hostname | awk 'BEGIN {} {print "const char * build_hostname = \""$$0"\";"} END {} ' >> src/version.c

#######################################
# Rules for building gx
####################################
OBJS = device_funcs.o parameters.o grids.o reductions.o reservoir.o grad_perp.o fields.o moments.o forcing.o grad_parallel.o grad_parallel_linked.o geometry.o hermite_transform.o laguerre_transform.o nca.o ncdf.o solver.o smith_par_closure.o closures.o linear.o nonlinear.o ts_sspx2.o ts_sspx3.o ts_rk2.o ts_rk3.o ts_rk4.o ts_k10.o ts_k2.o ts_g3.o diagnostics.o run_gx.o version.o trinity_interface.o

VMEC_GEO_OBJS = solver.o vmec_variables.o geometric_coefficients.o
VMEC_GEO_HEADERS = $(wildcard geometry_modules/vmec/include*.h)

obj/geo/%.o: %.cpp $(VMEC_GEO_HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCS) -I. -I geometry_modules/vmec/include

# main program
gx: obj/main.o libgx.a
	$(NVCC) -dlink $(NVCCFLAGS) -o obj/gx.o $< -L. -lgx $(LIBS) 
	$(CC) -o $@ obj/gx.o obj/main.o -L. -lgx $(LIBS) 
	@rm src/version.c

libgx.a: $(addprefix obj/, $(OBJS)) $(HEADERS) $(addprefix obj/geo/, $(VMEC_GEO_OBJS)) $(VMEC_GEO_HEADERS)
	ar -crs libgx.a $(addprefix obj/, $(OBJS)) $(addprefix obj/geo/, $(VMEC_GEO_OBJS))

libgx.so: $(addprefix obj/, $(OBJS)) $(HEADERS) $(addprefix obj/geo/, $(VMEC_GEO_OBJS)) $(VMEC_GEO_HEADERS)
	$(NVCC) -dlink $(NVCCFLAGS) -o obj/device.o $(addprefix obj/, $(OBJS)) $(addprefix obj/geo/, $(VMEC_GEO_OBJS))
	$(CC) -shared -o libgx.so obj/device.o $(addprefix obj/, $(OBJS)) $(addprefix obj/geo/, $(VMEC_GEO_OBJS))

geometry_modules/vmec/convert_VMEC_to_GX: obj/geo/main.o $(addprefix obj/geo/, $(VMEC_GEO_OBJS)) $(VMEC_GEO_HEADERS)
	$(CC) -o $@ $< $(addprefix obj/geo/, $(VMEC_GEO_OBJS)) $(LIBS)

all: gx

########################
# Cleaning up
########################

clean: 
	rm -rf obj/*.o obj/geo/*.o *~ libgx.a \#*

distclean: clean clean_tests
	rm -rf $(TARGET)

#########################
# Misc
#######################


test_make:
	@echo TARGET=    $(TARGET)
	@echo SDKDIR=    $(SDKDIR)
	@echo NVCC=      $(NVCC)
	@echo CFLAGS= 	 $(CFLAGS)
	@echo LDFLAGS= 	 $(LDFLAGS)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCC_FLAGS= $(NVCC_FLAGS)
	@echo CUDA_INC=  $(CUDA_INC)
	@echo CUDA_LIB=  $(CUDA_LIB)
	@echo CUTENSOR_LIB=  $(CUTENSOR_LIB)



