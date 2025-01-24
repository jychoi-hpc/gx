#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "device_funcs.h"
#include "diagnostics.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}

bool masked(int idx, int idy, int nx, int ny) {
  int ikx;
  if (idx < nx/2+1)
    ikx = idx;
  else
    ikx = idx-nx;
  if ( idx < nx           // index should be in range to be actively masked
    && idy < ny           // index should be in range to be actively masked
       && ( (idx==0 && idy==0) || idy > (ny-1)/3  || ikx > (nx-1)/3 || ikx < -(nx-1)/3 ))
    return true;
  else
    return false;
}

class TestMomentsG : public ::testing::Test {

protected:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    MPI_Comm mpcom = MPI_COMM_WORLD;
    MPI_Comm_rank(mpcom, &iproc);
    MPI_Comm_size(mpcom, &nprocs);

    int devid = 0; // This should be determined (optionally) on the command line
    int nGPUs = 0;
    cudaGetDeviceCount(&nGPUs);
    checkCuda(cudaSetDevice(iproc%nGPUs));
    cudaDeviceSynchronize();
    pars = new Parameters(iproc, nprocs, mpcom);
    pars->get_nml_vars("inputs/cyc_nl");
    pars->nx_in = 4;
    pars->ny_in = 4;
    pars->nz_in = 4;
    pars->nperiod = 1;
    pars->nspec_in = 2;
    pars->nm_in = 8;
    pars->nl_in = 4;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;
    pars->write_fluxes = false;
    pars->write_fields = false;
    pars->write_moms = false;

    grids = new Grids(pars);
    geo = new S_alpha_geo(pars,grids);
    diagnostics = new Diagnostics_GK(pars, grids, geo, nullptr, nullptr);
  }

  virtual void TearDown() {
    delete pars;
    delete geo;
    delete grids;
    delete diagnostics;
  }

  Parameters* pars;
  Grids *grids;
  Geometry* geo;
  Diagnostics* diagnostics;
  int iproc, nprocs;
};

TEST_F(TestMomentsG, InitConditions) {
  MomentsG *G;
  G = new MomentsG(pars, grids, 0);

  pars->init_single = false;
  pars->init_amp = .01;
  pars->ikpar_init = 2;
  pars->random_init = false;

  cuComplex* init_check = (cuComplex*) malloc(sizeof(cuComplex)*grids->NxNycNz);
  float *z_h = grids->z_h;

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index].x = 0.;
        init_check[index].y = 0.;
      }
    }
  }
  G->set_zero();

  srand(22);
  float samp;
  int idx;
  //      printf("Hacking the initial condition! \n");
  for(int i=0; i < 1 + (grids->Nx - 1)/3; i++) {
    for(int j=1; j < 1 + (grids->Ny - 1)/3; j++) {
      samp = pars->init_amp;
      float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
      float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
      for (int js=0; js < 2; js++) {
        if (i==0) {
          idx = i;
        } else {
          idx = (js==0) ? i : grids->Nx-i;
        }
        for(int k=0; k<grids->Nz; k++) {
          int index = j + grids->Nyc*(idx + grids->Nx*k);
          if (js == 0) {
    	init_check[index].x = ra;		init_check[index].y = rb;
          } else {
    	init_check[index].x = rb;		init_check[index].y = ra;
          }
          if (pars->ikpar_init < 0) {		
    	init_check[index].x *= (cos( -pars->ikpar_init    *z_h[k]/pars->Zp)
    			  + cos((-pars->ikpar_init+1.)*z_h[k]/pars->Zp));
    	init_check[index].y *= (cos( -pars->ikpar_init    *z_h[k]/pars->Zp)
    			  + cos((-pars->ikpar_init+1.)*z_h[k]/pars->Zp));
          } else {
    	init_check[index].x *= cos(pars->ikpar_init*z_h[k]/pars->Zp);
    	init_check[index].y *= cos(pars->ikpar_init*z_h[k]/pars->Zp);
          }
          	    //printf("init_h[%d] = (%e, %e) \n",index,init_h[index].x,init_h[index].y);
        }
      }
    }
  }

  // initialize upar
  pars->initf = inits::upar;
  G->initialConditions();

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        if(G->upar_ptr) {
	  EXPECT_FLOAT_EQ_D(&G->upar_ptr[index].x, init_check[index].x); 
          EXPECT_FLOAT_EQ_D(&G->upar_ptr[index].y, init_check[index].y);
	}
      }
    }
  }

  // initialize dens
  pars->initf = inits::density;
  G->initialConditions();

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        if(G->dens_ptr) {
          EXPECT_FLOAT_EQ_D(&G->dens_ptr[index].x, init_check[index].x);
          EXPECT_FLOAT_EQ_D(&G->dens_ptr[index].y, init_check[index].y);
	}
        if(G->upar_ptr) {
          EXPECT_FLOAT_EQ_D(&G->upar_ptr[index].x, init_check[index].x);
          EXPECT_FLOAT_EQ_D(&G->upar_ptr[index].y, init_check[index].y);
	}
        //EXPECT_FLOAT_EQ_D(&G->tpar_ptr[index].x, 0.);
        //EXPECT_FLOAT_EQ_D(&G->G(0,3)[index].x, 0.);
      }
    }
  }
 
  free(init_check);
  delete G;
}

TEST_F(TestMomentsG, SyncG)
{
  size_t size = grids->size_G;
  cuComplex* init = (cuComplex*) malloc(size);
  cuComplex* res = (cuComplex*) malloc(size);
  float* check = (float*) malloc(size);
  for(int i=0; i<grids->NxNycNz; i++) {
    for(int l=0; l<grids->Nl; l++) {
      for(int m_loc=-grids->m_ghost; m_loc<grids->Nm+grids->m_ghost; m_loc++) {
        int m = m_loc + grids->m_lo;
        int index = i + grids->NxNycNz*l + grids->Nl*grids->NxNycNz*(m_loc+grids->m_ghost);
        int fac = 1;
        if(m_loc < 0 || m_loc >= grids->Nm) fac = -1;
        init[index].x = fac*(i+grids->NxNycNz*l);
        init[index].y = fac*m;
	check[index] = m;
        //if(i=0) printf("GPU %d: f(%d, %d, %d) = (%d, %d)\n", grids->iproc, i, l, m_loc, (int) init[index].x, (int) init[index].y);
      }
    }
  }
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);
    CP_TO_GPU(G[is]->Gghost(), init, size);

    G[is]->sync();
    cudaStreamSynchronize(G[is]->syncStream);

    CP_TO_CPU(res, G[is]->Gghost(), size);
    cudaDeviceSynchronize();
    for(int i=0; i<grids->NxNycNz; i++) {
      for(int l=0; l<grids->Nl; l++) {
        for(int m_loc=-grids->m_ghost; m_loc<grids->Nm+grids->m_ghost; m_loc++) {
          int m = m_loc + grids->m_lo;
          int index = i + grids->NxNycNz*l + grids->Nl*grids->NxNycNz*(m_loc+grids->m_ghost);
          int fac = 1;
          if((grids->iproc_m-1 < 0 && m_loc<0) || (grids->iproc_m+1 >= grids->nprocs_m && m_loc>=grids->Nm)) fac = -1;
          EXPECT_FLOAT_EQ_D(&G[is]->Gghost()[index].x, fac*(i+grids->NxNycNz*l));
          EXPECT_FLOAT_EQ_D(&G[is]->Gghost()[index].y, fac*check[index]);
          //if(i=0) printf("GPU %d: g(%d, %d, %d) = (%d, %d) == (%d)\n", grids->iproc, i, l, m_loc, (int) res[index].x, (int) res[index].y, (int) (fac*check[index]));
        }
      }
    }
  }
  for(int is=0; is<grids->Nspecies; is++) {
    delete G[is];
  }
  free(G);
  free(init);
  free(res);
  free(check);
}

TEST_F(TestMomentsG, Restart) {
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);

  size_t size = grids->NxNycNz*grids->Nmoms*sizeof(cuComplex);
  cuComplex* init = (cuComplex*) malloc(size);
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);

    for(int i=0; i<grids->NxNycNz; i++) {
      for(int l=0; l<grids->Nl; l++) {
        for(int m_loc=0; m_loc<grids->Nm; m_loc++) {
          int m = m_loc + grids->m_lo;
          int index = i + grids->NxNycNz*l + grids->Nl*grids->NxNycNz*(m_loc);
          int fac = 1;
          if(m_loc < 0 || m_loc >= grids->Nm) fac = -1;
          init[index].x = (is_glob+1)*( i+grids->NxNycNz*l+grids->NxNycNz*grids->Nl*m );
          init[index].y = (is_glob+1)*m;
        }
      }
    }

    CP_TO_GPU(G[is]->G(), init, size);
  }

  double t = 0.1;
  diagnostics->restart_write(G, &t);

  t = -10.0;
  for(int is=0; is<grids->Nspecies; is++) {
    G[is]->set_zero();
    G[is]->restart_read(&t);

    int is_glob = is+grids->is_lo;
    for(int m_loc=0; m_loc<grids->Nm; m_loc++) {
      for(int l=0; l<grids->Nl; l++) {
        for(int k=0; k<grids->Nz; k++) {
          for (int i=0; i < grids->Nx; i++) {
            for (int j=0; j < grids->Nyc; j++) {
              int m = m_loc + grids->m_lo;
              unsigned int index = j + grids->Nyc *(i  + grids->Nx  *(k + grids->Nz*(l + grids->Nl*m_loc)));
              int fac = is_glob + 1;
              if(masked(i, j, grids->Nx, grids->Ny)) fac = 0.;
              EXPECT_FLOAT_EQ_D(&G[is]->G()[index].x, fac* ( j + grids->Nyc*(i+grids->Nx*k)+grids->NxNycNz*l+grids->NxNycNz*grids->Nl*m) );
              EXPECT_FLOAT_EQ_D(&G[is]->G()[index].y, fac*m);
            }
          }
        }
      }
    }
    EXPECT_FLOAT_EQ(t, 0.1);
  }

  for(int is=0; is<grids->Nspecies; is++) {
    delete G[is];
  }
  free(G);
  free(init);
}

