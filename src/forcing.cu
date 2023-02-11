#include "forcing.h"
#define GSINGLE <<< 1, 1 >>>

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

