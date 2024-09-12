#include "spectra_calc.h"

SpectraCalc::~SpectraCalc()
{
  cudaFree(data);
  free(tmp);
  free(cpu);
  delete field_reduce;
  delete moments_reduce;
}

void SpectraCalc::allocate()
{
  cudaMalloc (&data, sizeof(float) * N);
  tmp = (float*) malloc  (sizeof(float) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);
}

int SpectraCalc::define_nc_variable(string varstem, int nc_group, string description, bool append)
{
  int varid;
  if(append) {
    NC_ERR( nc_inq_varid(nc_group, (varstem + tag).c_str(), &varid) );
    NC_ERR( nc_var_par_access(nc_group, varid, NC_COLLECTIVE) );
  } else {
    NC_ERR( nc_def_var(nc_group, (varstem + tag).c_str(), NC_FLOAT, ndim, dims, &varid) );
    NC_ERR( nc_var_par_access(nc_group, varid, NC_COLLECTIVE) );
    NC_ERR( nc_put_att_text(nc_group, varid, "description", strlen(description.c_str()), description.c_str()) );
  }

  return varid;
}

void SpectraCalc::write(float *fullData, int varid, size_t time_index, int nc_group, bool isMoments, bool skip)
{
  if(isMoments) moments_reduce->Sum(fullData, data); 
  else field_reduce->Sum(fullData, data); 
  CP_TO_CPU(tmp, data, sizeof(float)*N);
  dealias_and_reorder(tmp, cpu);
  
  start[0] = time_index;
  if(skip) { 
    // sometimes we need to skip the write on a particular (set of) proc(s), 
    // but all procs still need to call nc_put_vara. so do an empty dummy write
    NC_ERR( nc_put_vara(nc_group, varid, dummy_start, dummy_count, cpu) );
  } else {
    NC_ERR( nc_put_vara(nc_group, varid, start, count, cpu) );
  }
}

SpectraCalc_st::SpectraCalc_st(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_st";
  reduced_modes = {'s'};
  ndim = 2;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nspecies;
  Nwrite = grids->Nspecies; 

  allocate();
}

SpectraCalc_kxst::SpectraCalc_kxst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxst";
  reduced_modes = {'x', 's'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nakx;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nx*grids->Nspecies;
  Nwrite = grids->Nakx*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxst::dealias_and_reorder(float *fold, float *fnew)
{
  for (int is = 0; is < grids_->Nspecies; is++) {
    int NK = grids_->Nakx/2;
    int NX = grids_->Nx; 
    int it = 0;
    int itp = it + NK;
    fnew[itp + is*grids_->Nakx] = fold[it + is*grids_->Nx];
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;

      fnew[itp + is*grids_->Nakx] = fold[it + is*grids_->Nx];
      fnew[itn + is*grids_->Nakx] = fold[itm + is*grids_->Nx];
    }
  }
}

SpectraCalc_kyst::SpectraCalc_kyst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kyst";
  reduced_modes = {'y', 's'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->ky;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Naky;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nyc*grids->Nspecies;
  Nwrite = grids->Naky*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kyst::dealias_and_reorder(float *fold, float *fnew)
{
  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {
      fnew[ik + is*grids_->Naky] = fold[ik + is*grids_->Nyc];
    }
  }
}

SpectraCalc_kxkyst::SpectraCalc_kxkyst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxkyst";
  reduced_modes = {'y', 'x', 's'};
  ndim = 4;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->ky;
  dims[3] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Naky;
  count[3] = grids->Nakx;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nx*grids->Nyc*grids->Nspecies;
  Nwrite = grids->Nakx*grids->Naky*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxkyst::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  for (int is = 0; is < grids_->Nspecies; is++) {
    int it = 0;
    int itp = it + NK;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
      int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
      fnew[Qp] = fold[Rp];
    }	
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;
      
      for (int ik = 0; ik < grids_->Naky; ik++) {

        int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
        int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

        int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
        int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

        fnew[Qp] = fold[Rp];
        fnew[Qn] = fold[Rm];
      }
    }
  }
}

SpectraCalc_kxkyzst::SpectraCalc_kxkyzst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxkyzst";
  reduced_modes = {'y', 'x', 'z', 's'};
  ndim = 5;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->z;
  dims[3] = nc_dims->ky;
  dims[4] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nz;
  count[3] = grids->Naky;
  count[4] = grids->Nakx;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nx*grids->Nyc*grids->Nz*grids->Nspecies;
  Nwrite = grids->Nakx*grids->Naky*grids->Nz*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxkyzst::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int iz = 0; iz < grids_->Nz; iz++) {
      int it = 0;
      int itp = it + NK;
      for (int ik = 0; ik < grids_->Naky; ik++) {
        int Qp = itp + grids_->Nakx*(ik + grids_->Naky*(iz + is*grids_->Nz));
        int Rp = ik  + grids_->Nyc*(it + grids_->Nx*(iz + is*grids_->Nz));
        fnew[Qp] = fold[Rp];
      }	
      for (int it = 1; it < NK+1; it++) {
        int itp = NK + it;
        int itn = NK - it;
        int itm = NX - it;
        
        for (int ik = 0; ik < grids_->Naky; ik++) {

          int Qp = itp + grids_->Nakx*(ik + grids_->Naky*(iz + is*grids_->Nz));
          int Rp = ik  + grids_->Nyc*(it + grids_->Nx*(iz + is*grids_->Nz));

          int Qn = itn + grids_->Nakx*(ik + grids_->Naky*(iz + is*grids_->Nz));
          int Rm = ik  + grids_->Nyc*(itm + grids_->Nx*(iz + is*grids_->Nz));

          fnew[Qp] = fold[Rp];
          fnew[Qn] = fold[Rm];
        }
      }
    }
  }
}

SpectraCalc_zst::SpectraCalc_zst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_zst";
  reduced_modes = {'z', 's'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->z;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nz;

  start[1] = grids->is_lo;

  field_reduce = new Reduction<float>(grids, field_species_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nz*grids->Nspecies;
  Nwrite = grids->Nz*grids->Nspecies; 

  allocate();
}

SpectraCalc_lst::SpectraCalc_lst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_lst";
  reduced_modes = {'l', 's'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->l;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nl;

  start[1] = grids->is_lo;

  field_reduce = nullptr;
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nl*grids->Nspecies;
  Nwrite = grids->Nl*grids->Nspecies; 

  allocate();
}

SpectraCalc_mst::SpectraCalc_mst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_mst";
  reduced_modes = {'m', 's'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->m;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nm;

  start[1] = grids->is_lo;
  start[2] = grids->m_lo;

  field_reduce = nullptr;
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nm*grids->Nspecies;
  Nwrite = grids->Nm*grids->Nspecies; 

  allocate();
}

SpectraCalc_lmst::SpectraCalc_lmst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_lmst";
  reduced_modes = {'l', 'm', 's'};
  ndim = 4;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->m;
  dims[3] = nc_dims->l;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Nm;
  count[3] = grids->Nl;

  start[1] = grids->is_lo;
  start[2] = grids->m_lo;

  field_reduce = nullptr;
  moments_reduce = new Reduction<float>(grids, moment_species_modes, reduced_modes);

  N = grids->Nl*grids->Nm*grids->Nspecies;
  Nwrite = grids->Nl*grids->Nm*grids->Nspecies; 

  allocate();
}

SpectraCalc_t::SpectraCalc_t(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_t";
  reduced_modes = {};
  ndim = 1;

  dims[0] = nc_dims->time;

  count[0] = 1; // each write is a single time slice

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = 1;
  Nwrite = 1;

  allocate();
}

SpectraCalc_kxt::SpectraCalc_kxt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxt";
  reduced_modes = {'x'};
  ndim = 2;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nakx;

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = grids->Nx;
  Nwrite = grids->Nakx; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxt::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  int it = 0;
  int itp = it + NK;
  fnew[itp] = fold[it];
  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = NX - it;

    fnew[itp] = fold[it];
    fnew[itn] = fold[itm];
  }
}

SpectraCalc_kyt::SpectraCalc_kyt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kyt";
  reduced_modes = {'y'};
  ndim = 2;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->ky;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = grids->Nyc;
  Nwrite = grids->Naky; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kyt::dealias_and_reorder(float *fold, float *fnew)
{
  for (int ik = 0; ik < grids_->Naky; ik++) {
    fnew[ik] = fold[ik];
  }
}

SpectraCalc_kxkyt::SpectraCalc_kxkyt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxkyt";
  reduced_modes = {'y', 'x'};
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->ky;
  dims[2] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;
  count[2] = grids->Nakx;

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = grids->Nx*grids->Nyc;
  Nwrite = grids->Nakx*grids->Naky; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxkyt::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  int it = 0;
  int itp = it + NK;
  for (int ik = 0; ik < grids_->Naky; ik++) {
    int Qp = itp + ik*grids_->Nakx;
    int Rp = ik  + it*grids_->Nyc;
    fnew[Qp] = fold[Rp];
  }	
  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = NX - it;
    
    for (int ik = 0; ik < grids_->Naky; ik++) {

      int Qp = itp + ik*grids_->Nakx;
      int Rp = ik  + it*grids_->Nyc;

      int Qn = itn + ik *grids_->Nakx;
      int Rm = ik  + itm*grids_->Nyc;

      fnew[Qp] = fold[Rp];
      fnew[Qn] = fold[Rm];
    }
  }
}

SpectraCalc_zt::SpectraCalc_zt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_zt";
  reduced_modes = {'z'};
  ndim = 2;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->z;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nz;

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = grids->Nz;
  Nwrite = grids->Nz;

  allocate();
}

SpectraCalc_kxkyzt::SpectraCalc_kxkyzt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxkyzt";
  reduced_modes = {'y', 'x', 'z'};
  ndim = 4;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->z;
  dims[2] = nc_dims->ky;
  dims[3] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nz;
  count[2] = grids->Naky;
  count[3] = grids->Nakx;

  field_reduce = new Reduction<float>(grids, field_modes, reduced_modes);
  moments_reduce = new Reduction<float>(grids, moment_modes, reduced_modes);

  N = grids->Nx*grids->Nyc*grids->Nz;
  Nwrite = grids->Nakx*grids->Naky*grids->Nz; // only write de-aliased modes

  allocate();
}

void SpectraCalc_kxkyzt::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  for (int iz = 0; iz < grids_->Nz; iz++) {
    int it = 0;
    int itp = it + NK;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + grids_->Nakx*(ik + grids_->Naky*(iz));
      int Rp = ik  + grids_->Nyc*(it + grids_->Nx*(iz));
      fnew[Qp] = fold[Rp];
    }	
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;
      
      for (int ik = 0; ik < grids_->Naky; ik++) {

        int Qp = itp + grids_->Nakx*(ik + grids_->Naky*(iz));
        int Rp = ik  + grids_->Nyc*(it + grids_->Nx*(iz));

        int Qn = itn + grids_->Nakx*(ik + grids_->Naky*(iz));
        int Rm = ik  + grids_->Nyc*(itm + grids_->Nx*(iz));

        fnew[Qp] = fold[Rp];
        fnew[Qn] = fold[Rm];
      }
    }
  }
}
