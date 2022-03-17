#include "nca.h"

nca::nca(int N, int Nwrite) :
  N_(N), Nwrite_(Nwrite)
{
  data = nullptr;  cpu = nullptr;  tmp = nullptr; z_tmp = nullptr; tmp_d = nullptr;
  write = false;  
  write_v_time = false;
  xydata = false;
  xdata = false;
  scalar = false;
  mdy = false;
  dy  = false;
  dx  = false;
  d2x = false;
  all = false; // true to keep the zonal component in xy plots
  adj = 1.0;
  time_start[0] = 0;
  time_start[1] = 0;
  time_start[2] = 0;
  time_start[3] = 0;
  time_start[4] = 0;
  time_start[5] = 0;

  time_count[0] = 1;
  
  if (N == 0) return;
      
  if (N > 0) {
    cudaMalloc (&data, sizeof(float) * N);
    if (Nwrite > 0) {
      cudaMalloc      (&tmp_d, sizeof(float) * Nwrite);  // not needed for spectra
      tmp = (float*) malloc  (sizeof(float) * N);
      cpu = (float*) malloc  (sizeof(float) * Nwrite);
    } else {
      cpu = (float*) malloc  (sizeof(float) * N);
    }
  } else { // omega only
    N = -N;
    if (Nwrite > 0) {
      z_tmp = (cuComplex*) malloc  (sizeof(cuComplex) * N);
      cpu = (float*) malloc  (sizeof(float) * Nwrite);
    }
  }  
}
nca::~nca() {
  if (data)  cudaFree     ( data   );
  if (tmp_d) cudaFree     ( tmp_d  );
  if (tmp)   free ( tmp    );
  if (cpu)   free ( cpu    );
  if (z_tmp) free ( z_tmp  );
}
void nca::increment_ts(void) {time_start[0] += 1;}
