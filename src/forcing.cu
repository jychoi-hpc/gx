#include "forcing.h"
#include "get_error.h"
#include "device_funcs.h"
#include "cuda_constants.h"

void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt);

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
  
  if (pars_->stirf == DENS) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == UPAR) {stirring_kernel<<<1,1>>> (rf,           G->upar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(6.0), G->qpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPRP) {stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPRP) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->qprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPRP) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPAR) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}
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

  if (pars_->stirf == DENS) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == UPAR) {stirring_kernel<<<1,1>>> (rf,           G->upar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(6.0), G->qpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPRP) {stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPRP) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->qprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPRP) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPAR) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}
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

  if (pars_->stirf == DENS) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == UPAR) {stirring_kernel<<<1,1>>> (rf,           G->upar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPAR) {stirring_kernel<<<1,1>>> (rf*sqrt(6.0), G->qpar_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == TPRP) {stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == QPRP) {stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->qprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPRP) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf,           G->tprp_ptr[0], pars_->forcing_index);}
  if (pars_->stirf == PPAR) {stirring_kernel<<<1,1>>> (rf,           G->dens_ptr[0], pars_->forcing_index);
                             stirring_kernel<<<1,1>>> (rf*sqrt(2.0), G->tpar_ptr[0], pars_->forcing_index);}

  stirring_done = true;
}

