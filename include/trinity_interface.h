#pragma once
#include "parameters.h"

struct trin_parameters_struct;
struct trin_fluxes_struct;

struct trin_parameters_struct {
   int mpirank;
   int restart;
   int nstep;
   int navg;
   double end_time;
   int job_id;
   int trinity_timestep;
   int trinity_iteration;
   int trinity_conv_count;
   /* Name of gryfx/gryffin input file*/
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
};

struct trin_fluxes_struct {
   double pflux[20];
   double qflux[20];
   double heat[20];
};

extern "C"
void gx_get_fluxes_(struct trin_parameters_struct * tpars, struct trin_fluxes_struct* tfluxes, char * namelistFile, int mpcom);

void set_from_trinity(Parameters *pars, trin_parameters_struct *tpars);
void copy_fluxes_to_trinity(Parameters *pars, trin_fluxes_struct *tfluxes);
