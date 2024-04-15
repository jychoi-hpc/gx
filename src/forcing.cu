#include "forcing.h"
#include <cuda_runtime.h>
#define GSINGLE <<< 1, 1 >>>

void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt);
void heli_generate_random_numbers(float *random_real, float *random_imag, float pos_forcing_amp_, float neg_forcing_amp_, float dt);
/* The following knobs in forcing_knobs need to be set to turn on forcing:
forcing_init = T
forcing_amp = 1
forcing_type = "Kz" or "KzImpulse"
*/

// Set forcing_type to "Kz"

KzForcing::KzForcing(Parameters *pars) : pars_(pars)
{
  forcing_amp_ = pars_->forcing_amp;
  printf("forcing_amp = %f \n", forcing_amp_);
}

KzForcing::~KzForcing()
{
}

// Langevin forcing done at the end of each timestep in timestepper's advance method
void KzForcing::stir(MomentsG *G) {
  float random_real, random_imag;

  generate_random_numbers (&random_real, &random_imag, forcing_amp_, pars_->dt);

  rf.x = random_real;
  //  rf.y = random_imag;
  rf.y = 0.;
  
  switch (pars_->stirf)
    {
    case stirs::density : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index); break;
    case stirs::upar    : stirring_kernel GSINGLE (rf,           G->upar_ptr, pars_->forcing_index); break;
    case stirs::tpar    : stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::tperp   : stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    case stirs::qpar    : stirring_kernel GSINGLE (rf*sqrt(6.0), G->qpar_ptr, pars_->forcing_index); break;
    case stirs::qperp   : stirring_kernel GSINGLE (rf*sqrt(2.0), G->qprp_ptr, pars_->forcing_index); break;
    case stirs::ppar    : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::pperp   : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    }
                             
}

HeliInjForcing::HeliInjForcing(Parameters *pars, Grids *grids) : pars_(pars), grids_(grids)
{
  pos_forcing_amp_ = pars_->pos_forcing_amp;
  neg_forcing_amp_ = pars_->neg_forcing_amp;
  k2min = pars_->forcing_k2min;
  k2max = pars_->forcing_k2max;
  kz = pars_->forcing_kz;
  Nz = grids_->Nz;
}

HeliInjForcing::~HeliInjForcing()
{
}

void HeliInjForcing::stir(MomentsG *G) {
  float random_real, random_imag;
  int kx, ky;
  kx = rand() % (k2max + 1);
  ky = rand() % (k2max + 1);
  //std::cout << "Perturbed mode: (" << kx << ", " << ky << ")" << std::endl;
  heli_generate_random_numbers (&random_real, &random_imag, pos_forcing_amp_, neg_forcing_amp_, pars_->dt);
  rf.x = random_real;
  rf.y = random_imag;
  switch (pars_->stirf)
  {
   case stirs::density : kz_stirring_kernel <<<Nz,1>>> (rf, G->dens_ptr, kx, ky, kz); break;
   case stirs::upar    : kz_stirring_kernel <<<Nz,1>>> (rf, G->upar_ptr, kx, ky, kz); break;
   case stirs::tpar    : kz_stirring_kernel <<<Nz,1>>> (rf * sqrt(2.0), G->tpar_ptr, kx, ky, kz); break;
   case stirs::tperp   : kz_stirring_kernel <<<Nz,1>>> (rf, G->tprp_ptr, kx, ky, kz); break;
   case stirs::qpar    : kz_stirring_kernel <<<Nz,1>>> (rf * sqrt(6.0), G->qpar_ptr, kx, ky, kz); break;
   case stirs::qperp   : kz_stirring_kernel <<<Nz,1>>> (rf * sqrt(2.0), G->qprp_ptr, kx, ky, kz); break;
   case stirs::ppar    : kz_stirring_kernel <<<Nz,1>>> (rf, G->dens_ptr, kx, ky, kz);
                      kz_stirring_kernel <<<Nz,1>>> (rf * sqrt(2.0), G->tpar_ptr, kx, ky, kz); break;
   case stirs::pperp   : kz_stirring_kernel <<<Nz,1>>> (rf, G->dens_ptr, kx, ky, kz);
                      kz_stirring_kernel <<<Nz,1>>> (rf, G->tprp_ptr, kx, ky, kz); break;
  }
}





genForcing::genForcing(Parameters *pars) : pars_(pars)
{
  forcing_amp_ = pars_->forcing_amp;
}

genForcing::~genForcing()
{
}

void genForcing::stir(MomentsG *G) {
  float random_real, random_imag;
  generate_random_numbers (&random_real, &random_imag, forcing_amp_, pars_->dt);

  rf.x = random_real;
  rf.y = random_imag;

  switch (pars_->stirf)
    {
    case stirs::density : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index); break;
    case stirs::upar    : stirring_kernel GSINGLE (rf,           G->upar_ptr, pars_->forcing_index); break;
    case stirs::tpar    : stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::tperp   : stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    case stirs::qpar    : stirring_kernel GSINGLE (rf*sqrt(6.0), G->qpar_ptr, pars_->forcing_index); break;
    case stirs::qperp   : stirring_kernel GSINGLE (rf*sqrt(2.0), G->qprp_ptr, pars_->forcing_index); break;
    case stirs::ppar    : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::pperp   : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    }
}

// Set forcing_type to "KzImpulse"
KzForcingImpulse::KzForcingImpulse(Parameters *pars) : KzForcing(pars)
{
  stirring_done = false;
}

// Langevin forcing done once at the first timestep
void KzForcingImpulse::stir(MomentsG *G) {
  if(stirring_done) {
    return;
  }

  float random_real, random_imag;
  generate_random_numbers(&random_real, &random_imag, forcing_amp_, pars_->dt);

  rf.x = random_real;
  rf.y = random_imag;

  switch (pars_->stirf)
    {                   
    case stirs::density : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index); break;
    case stirs::upar    : stirring_kernel GSINGLE (rf,           G->upar_ptr, pars_->forcing_index); break;
    case stirs::tpar    : stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::tperp   : stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    case stirs::qpar    : stirring_kernel GSINGLE (rf*sqrt(6.0), G->qpar_ptr, pars_->forcing_index); break;
    case stirs::qperp   : stirring_kernel GSINGLE (rf*sqrt(2.0), G->qprp_ptr, pars_->forcing_index); break;
    case stirs::ppar    : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf*sqrt(2.0), G->tpar_ptr, pars_->forcing_index); break;
    case stirs::pperp   : stirring_kernel GSINGLE (rf,           G->dens_ptr, pars_->forcing_index);
                          stirring_kernel GSINGLE (rf,           G->tprp_ptr, pars_->forcing_index); break;
    }

  stirring_done = true;
}

void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt) {

  // Box-Muller transform to generate random normal variables
  float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );

  // dt term in timestepper scheme accounted for in amp
  float amp = sqrt(abs(forcing_amp_*dt*log(ran_amp)));
  float phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

  *random_real = amp*cos(phase);
  *random_imag = amp*sin(phase);
}

void heli_generate_random_numbers(float *random_real, float *random_imag, float pos_forcing_amp_, float neg_forcing_amp_, float dt)
{ 
  // dt term in timestepper scheme accounted for in amp
  float phase = M_PI * (2.0 * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0) - 1.0);
  
  float amp;  // Variable to hold the amplitude

  float ran_amp;

  if (phase <= M_PI / 2.0 && phase >= -M_PI / 2.0) {
    ran_amp = pos_forcing_amp_;
  } else {
    ran_amp = neg_forcing_amp_;
  }
  amp = sqrt(abs(ran_amp*dt));
  *random_real = amp*cos(phase);
  *random_imag = amp*sin(phase);
}
