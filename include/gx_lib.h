#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#ifdef USE_MPI
#include "mpi.h"
#else
typedef int MPI_Comm;
#endif

#define DEBUGPRINT(_fmt, ...)  if (pars->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define DEBUG_PRINT(_fmt, ...)  if (pars_->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define CP_ON_GPU(to, from, isize) cudaMemcpy(to, from, isize, cudaMemcpyDeviceToDevice)
#define CP_TO_GPU(gpu, cpu, isize) cudaMemcpy(gpu, cpu, isize, cudaMemcpyHostToDevice)
#define CP_TO_CPU(cpu, gpu, isize) cudaMemcpy(cpu, gpu, isize, cudaMemcpyDeviceToHost)

#define CUDA_DEBUG(_fmt, ...) if (pars->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__, cudaGetErrorString(cudaGetLastError()))

#define ERR(e) {printf("Error: %s. See file: %s, line %d\n", nc_strerror(e),__FILE__,__LINE__); exit(2);}


#pragma once

//char *runname;

/* These must be in the same order that they appear in
 * gs2_gx_zonal.f90*/
struct external_parameters_struct {
  int mpirank;
  int restart;
  int nstep;
  int navg;
  double end_time;
  int job_id;
  int trinity_timestep;
  int trinity_iteration;
  int trinity_conv_count;
  /* Name of gx/gryffin input file*/
  /*char input_file[1000];*/
  /*Base geometry parameters - not currently set by trinity 
    !See geometry.f90*/
  int equilibrium_type;
  /*char eqfile[800];*/
  int irho;
  double rhoc;
  int bishop;
  int nperiod;
  int ntheta;
  
  /* Miller parameters*/
  double rgeo_lcfs;
  double rgeo_local;
  double akappa;
  double akappri;
  double tri;
  double tripri;
  double shift;
  double qinp;
  double shat;
  double asym;
  double asympri;
  
  /* Circular flux surfaces*/
  double eps;
  
  /* Other geometry parameters - Bishop/Greene & Chance*/
  double beta_prime_input;
  double s_hat_input;
  
  /*Flow shear*/
  double g_exb;
  
  /* Species parameters... I think allowing 20 species should be enough!*/
  int ntspec;
  double dens[20];
  double temp[20];
  double fprim[20];
  double tprim[20];
  double nu[20];
  
  void * pars_address;
};

struct gx_outputs_struct {
  double pflux[20];
  double qflux[20];
  double heat[20];
  double dvdrho;
  double grho;
};

void gx_get_default_parameters_(struct external_parameters_struct *, char *run_name, MPI_Comm mpcom);
extern "C"
void gx_get_fluxes_(struct external_parameters_struct *, struct gx_outputs_struct*, MPI_Comm mpcom);

void gx_main(int argc, char* argv[], MPI_Comm mpcom);


