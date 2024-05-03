#include "fields.h"
#include "get_error.h"
#include "grad_perp.h"
#include <cmath> // Include cmath for mathematical functions
#include <iostream>

Fields::Fields(Parameters* pars, Grids* grids) :
  size_(sizeof(cuComplex)*grids->NxNycNz), sizeReal_(sizeof(float)*grids->NxNyNz), N(grids->NxNycNz), pars_(pars), grids_(grids),
  phi(nullptr), phi_h(nullptr), apar(nullptr), apar_h(nullptr), bpar(nullptr), bpar_h(nullptr),
  apar_ext(nullptr), apar_ext_h(nullptr), apar_ext_realspace_h(nullptr), apar_ext_realspace(nullptr),
  ne(nullptr), ne_h(nullptr), ue(nullptr), ue_h(nullptr), Te(nullptr), Te_h(nullptr)
{
  checkCuda(cudaMalloc((void**) &phi, size_));

  int nn = grids->NxNycNz; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
  setval <<< nb, nt >>> (phi, zero, nn); // should replace this with cudaMemset

  //  cudaMemset(phi, 0., size_);
  bool debug = pars->debug;

  phi_h = (cuComplex*) malloc(size_);
  DEBUGPRINT("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  checkCuda(cudaMalloc((void**) &apar, size_));
  DEBUGPRINT("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  setval <<< nb, nt >>> (apar, zero, nn);
  
  checkCuda(cudaMalloc((void**) &apar_ext, size_));
  if(debug) printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  checkCuda(cudaMalloc((void**) &apar_ext_realspace, sizeReal_));
  
  setval <<< nb, nt >>> (apar_ext, zero, nn);
  
  apar_h = (cuComplex*) malloc(size_);
  apar_ext_realspace_h = (float*) malloc(sizeReal_);
  apar_ext_h = (cuComplex*) malloc(size_);
    
  checkCuda(cudaMalloc((void**) &bpar, size_));
  if(debug) printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  setval <<< nb, nt >>> (bpar, zero, nn);
  
  bpar_h = (cuComplex*) malloc(size_);
  
  //if (pars_->beta > 0. || pars_->krehm) {
  //  if (!pars_->krehm) {
  //    checkCuda(cudaMalloc((void**) &ne, size_));
  //    printf("Allocated ne array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(ne, 0., size_); setval <<< nb, nt >>> (ne, zero, nn);
  //    
  //    ne_h = (cuComplex*) malloc(size_);
  //    
  //    checkCuda(cudaMalloc((void**) &ue, size_));
  //    printf("Allocated ue array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(ue, 0., size_); setval <<< nb, nt >>> (ue, zero, nn);
  //    
  //    ue_h = (cuComplex*) malloc(size_);
  //    
  //    checkCuda(cudaMalloc((void**) &Te, size_));
  //    printf("Allocated Te array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(Te, 0., size_); setval <<< nb, nt >>> (Te, zero, nn);
  //    
  //    Te_h = (cuComplex*) malloc(size_);
  //  }
  //}

  if (pars_->harris_sheet && !pars_->restart) {

    assert((pars_->fapar > 0.) && "Harris sheet equilibrium requires setting fapar = 1.0");
    
    int nBatch = grids_->Nz;
    GradPerp * grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz);
    
    //set up harris sheet in real space   
    for(int idz = 0; idz < grids_->Nz; idz++) {
      for(int idx = 0; idx < grids_->Nx; idx++) {
        for(int idy = 0; idy < grids_->Ny; idy++) {
           float x = grids_->x_h[idx];
	   float xn = x/pars_->x0;
	   float pi = M_PI;
	   
	   int index = idy + idx * grids_->Ny + idz * grids_->NxNy;
	   float A0 = 1.29904; // this value makes B_ext_max = 1
	   apar_ext_realspace_h[index] = A0 / pow(cosh(xn - pi), 2)
	     * (
		( pow( tanh(xn), 2) + pow( tanh(xn - 2*pi), 2) -  pow(tanh(2*pi), 2) )
		/
		( 2*pow( tanh(pi), 2) - pow(tanh( 2*pi), 2) )
		);
	}
      }
    }

    //copy apar_ext to GPU and do Fourier transformation
    CP_TO_GPU(apar_ext_realspace, apar_ext_realspace_h, sizeof(float) * grids_->NxNyNz); 
    grad_perp->R2C(apar_ext_realspace, apar_ext, true);
    
    delete grad_perp;

    //debug part

    //grad_perp->qvar(apar_ext_realspace, grids_->NxNyNz); 
    //CP_TO_CPU(apar_ext_h, apar_ext, sizeof(cuComplex) * grids_->NxNycNz);
  //  for(int idz = 0; idz < grids_->Nz; idz++){
  //    for(int idx = 0; idx < grids_->Nx; idx++){
  //      for(int idy = 0; idy < grids_->Nyc; idy++){
  //         unsigned int idxyz = idy + grids_->Nyc *(idx + grids_->Nx*idz); 
  //         printf("idxyz%d:",idxyz);
  //         printf("%f\n",apar_ext_h[idxyz].x);
  //      }
  //    }
  //  }
    //grad_perp->qvar(apar_ext, grids_->NxNycNz); 
  }
  if (pars_->periodic_equilibrium && !pars_->restart) {
    int nBatch = grids_->Nz;
    GradPerp * grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz);

    //set up periodic Apar in real space   
    for(int idz = 0; idz < grids_->Nz; idz++) {
      for(int idx = 0; idx < grids_->Nx; idx++) {
        for(int idy = 0; idy < grids_->Ny; idy++) {
           float x = grids_->x_h[idx];
           float y = grids_->y_h[idy];
           int index = idy + idx * grids_->Ny + idz * grids_->NxNy;
           float A0 = 0.25; // this was recommended by Lucio
           apar_ext_realspace_h[index] = A0*cos(pars_->k0 * (x - M_PI*pars_->x0))*cos(pars_->k0 * (y-M_PI*pars_->y0));
        }
      }
    }

    //copy apar_ext to GPU and do Fourier transformation
    CP_TO_GPU(apar_ext_realspace, apar_ext_realspace_h, sizeof(float) * grids_->NxNyNz);
    grad_perp->R2C(apar_ext_realspace, apar_ext, true);

    delete grad_perp;

    //debug part

    //grad_perp->qvar(apar_ext_realspace, grids_->NxNyNz); 
    //CP_TO_CPU(apar_ext_h, apar_ext, sizeof(cuComplex) * grids_->NxNycNz);
  //  for(int idz = 0; idz < grids_->Nz; idz++){
  //    for(int idx = 0; idx < grids_->Nx; idx++){
  //      for(int idy = 0; idy < grids_->Nyc; idy++){
  //         unsigned int idxyz = idy + grids_->Nyc *(idx + grids_->Nx*idz); 
  //         printf("idxyz%d:",idxyz);
  //         printf("%f\n",apar_ext_h[idxyz].x);
  //      }
  //    }
  //  }
    //grad_perp->qvar(apar_ext, grids_->NxNycNz); 
  }
  if (pars_->island_coalesce && !pars_->restart) {
    int nBatch = grids_->Nz;
    GradPerp * grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz);

    for(int idz = 0; idz < grids_->Nz; idz++) {
      for(int idx = 0; idx < grids_->Nx; idx++) {
        for(int idy = 0; idy < grids_->Ny; idy++) {
           float x = grids_->x_h[idx];
           float y = grids_->y_h[idy];
           int index = idy + idx * grids_->Ny + idz * grids_->NxNy;

           //float B0 = 0.5;
	   //float lambda = 0.5;
	   //float epsilon = 0.4;
           //apar_ext_realspace_h[index] = B0*lambda*log(epsilon * cos((x - M_PI*pars_->x0)/lambda) + cosh((y - M_PI*pars_->y0)/lambda)) - 0.1*B0*lambda*cos((x - M_PI*pars_->x0)/lambda) * cos((y - M_PI*pars_->y0)/lambda);
	   float A0 = 0.7772;
	   float Lx = 2*M_PI*pars_->x0;
           float Ly = 2*M_PI*pars_->y0;
	   apar_ext_realspace_h[index] = A0*exp(-pow((2.0*M_PI*1.5/Lx*(x-0.27*Lx)),2) - pow((2.0*M_PI*1.5/Ly*(y-0.5*Ly)),2)) + A0*exp(-pow((2.0*M_PI*1.5/Lx*(x-0.73*Lx)),2)-pow((2.0*M_PI*1.5/Ly*(y-0.5*Ly)),2));
           
           //apar_ext_realspace_h[index] = A0*exp(-pow((2.0*M_PI*1.5/Lx*(x-0.26*Lx)),2) - pow((2.0*M_PI*1.5/Ly*(y-0.5*Ly)),2)) + A0*exp(-pow((2.0*M_PI*1.5/Lx*(x-0.74*Lx)),2)-pow((2.0*M_PI*1.5/Ly*(y-0.5*Ly)),2)) + 0.1*A0*sin(x-0.5*Lx)*sin(y-0.5*Ly);
	   //apar_ext_realspace_h[index] = 0.1 * A0 * sin(x-0.5*Lx) * sin(y-0.5*Ly);
        }
      }
    }
    
    CP_TO_GPU(apar_ext_realspace, apar_ext_realspace_h, sizeof(float) * grids_->NxNyNz);
    grad_perp->R2C(apar_ext_realspace, apar_ext, true);

    delete grad_perp;
  }
  if (pars_->random_gaussian && !pars_->restart) {
     std::cout << "Random Gaussian being placed" << std::endl;
     int NKX = grids_->Nx/2;
     int Nx = grids_->Nx;
     int Nyc = grids_->Nyc;
     int Nz = grids_->Nz;
     //for (int idz = 0; idz < Nz; idz++) {
     for (int ikx = 0; ikx <= Nx; ikx++) {
         int kx = (ikx > NKX ? ikx-Nx : ikx);
       for (int iky = 0; iky < Nyc; iky++){
         //int kx = (ikx > NKX ? ikx-Nx : ikx);
         int ky = iky;
         float phase = M_PI * (2.0 * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 1.0);
	 
	 int kmax2 = 225;
	 int kmin2 = 169;
	 float kmax4 = static_cast<float>(kmax2*kmax2);
	 float kmin4 = static_cast<float>(kmin2*kmin2);
	 //float A0 = 0.01*static_cast<float>(rand())/static_cast<float>(RAND_MAX);
	 float A0 = 0.025;
	 //float A0 = 0.05;

         for (int idz = 0; idz < Nz; idz++){
              //int ky = iky;
              unsigned int idxyz = iky + ikx * Nyc + Nx * Nyc * idz;

              //int kc = pars_->kc;
              //float kc2 = static_cast<float>(kc*kc);

              //float k2 = static_cast<float>(kx*kx+ky*ky);
	      int k2_int = kx*kx + ky*ky;
	      float z_coord = static_cast<float>(idz) / static_cast<float>(Nz);
	      //float z_mode = (cos(2.0 * M_PI * z_coord) + cos(2.0 * M_PI * 2.0 * z_coord) + cos(2.0 * M_PI * 3.0 * z_coord) + cos(2.0 * M_PI * 4.0 * z_coord) + cos(2.0 * M_PI * 5.0 * z_coord) + cos(2.0 * M_PI * 6.0 * z_coord) )/6.0;
	      //float z_mode = cos(2.0 * M_PI * z_coord);
	      float z_mode = 1.0;

	      //int kmax2 = 256;
	      //int kmin2 = 169;
	      //float kmax4 = static_cast<float>(kmax2*kmax2);
	      //float kmin4 = static_cast<float>(kmin2*kmin2);

	      //float A0 = 0.7772;
	      //float A0 = 50.0/(kmax4-kmin4);
	      //float A0 = static_cast<float>(rand())*50.0/(kmax4-kmin4)/static_cast<float>(RAND_MAX);
              //float phase = M_PI * (2.0 * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 1.0);
	      //float phase = 0.0;

              if ( abs(kx) == 10 && abs(ky) == 10) {
              //if ( k2_int <= kmax2 && k2_int >= kmin2) {
                //printf("kx:%d  ",kx);
                //printf("ky:%d  ",ky);
		//printf("k2:%d  ",k2_int);
		//printf("A0:%f  ",A0);
		//printf("\n");
                apar_ext_h[idxyz].x = A0*cos(phase)*z_mode;
                apar_ext_h[idxyz].y = A0*sin(phase)*z_mode;
              }
	    }
          }
        }
     CP_TO_GPU(apar_ext, apar_ext_h, size_);
  }
  if (pars_->gaussian_tube && !pars_->restart) {
    int nBatch = grids_->Nz;
    GradPerp * grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz);
    
    //set up Gaussian tube in real space   
    for(int idz = 0; idz < grids_->Nz; idz++) {
      for(int idx = 0; idx < grids_->Nx; idx++) {
        for(int idy = 0; idy < grids_->Ny; idy++) {
           float x = grids_->x_h[idx];
           float y = grids_->y_h[idy];
           int index = idy + idx * grids_->Ny + idz * grids_->NxNy;
           float A0 = 0.5829; // this value makes B_eq_max = 1
           apar_ext_realspace_h[index] = A0*exp(-pow((x - M_PI*pars_->x0)/(M_PI*pars_->x0),2)-pow((y - M_PI*pars_->y0)/(M_PI*pars_->y0),2)); // Need to multiply x0, y0 by 2pi. Shift Gaussian to the center, decrease width by 2.
        }
      }
    }   
    //copy apar_ext to GPU and do Fourier transformation
    CP_TO_GPU(apar_ext_realspace, apar_ext_realspace_h, sizeof(float) * grids_->NxNyNz); 
    grad_perp->R2C(apar_ext_realspace, apar_ext, true);
    
    delete grad_perp;
  }
  }


Fields::~Fields() {
  if (phi)     cudaFree(phi);
  if (apar)    cudaFree(apar);
  if (bpar)    cudaFree(bpar);

  if (phi_h)   free(phi_h);
  if (apar_h)  free(apar_h);
  if (bpar_h)  free(bpar_h);

  if (ne)      cudaFree(ne);
  if (ue)      cudaFree(ue);
  if (Te)      cudaFree(Te);

  if (ne_h)    free(ne_h);
  if (ue_h)    free(ue_h);
  if (Te_h)    free(Te_h);
  
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phi_h, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phi_h[j].x, phi_h[j].y);
  printf("\n");
}

void Fields::print_apar(void)
{
  CP_TO_CPU(apar_h, apar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("apar(%d) = (%e, %e) \n",j, apar_h[j].x, apar_h[j].y);
  printf("\n");
}

void Fields::print_bpar(void)
{
  CP_TO_CPU(bpar_h, bpar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("bpar(%d) = (%e, %e) \n",j, bpar_h[j].x, bpar_h[j].y);
  printf("\n");
}

void Fields::rescale(float * phi_max) {
  int nn1 = grids_->NxNyc; int nt1 = min(nn1, 32); int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nz;    int nt2 = min(nn2, 16); int nb2 = 1 + (nn2-1)/nt2;
  dim3 dB, dG;
  dB = dim3(nt1, nt2, 1);
  dG = dim3(nb1, nb2, 1);
  rescale_kernel <<< dG, dB >>> (phi, phi_max, 1);
  if(pars_->fapar > 0.) rescale_kernel <<< dG, dB >>> (apar,  phi_max, 1);
}
