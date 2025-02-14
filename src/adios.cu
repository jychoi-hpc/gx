#include "adios.h"

Adios::Adios(Parameters* pars, Grids* grids, std::string prefix, int dims, Geometry *geo, std::string engine): engine_(engine),
	pars_(pars), grids_(grids), geo_(geo),
	shape(dims, static_cast<unsigned int>(0)),
	start(dims, static_cast<unsigned int>(0)),
	count(dims, static_cast<unsigned int>(0)),
	adios_(pars->mpcom)
{
  io = adios_.DeclareIO("Write"+prefix);
  io.SetEngine(engine);
  std::string fname(pars_->run_name);
  fname += prefix;
  writer = io.Open(fname, adios2::Mode::Write);

  defineGridVar();
  defineGeometryVar();
}

void Adios::defineGridVar()
{
	io.DefineAttribute<float>("Grids/theta", grids_->z_h, grids_->Nz);
	timeVar = io.DefineVariable<double>("time");
}

void Adios::write_time(double time)
{
	writer.Put(timeVar, time);
}

void Adios::defineGeometryVar()
{
	io.DefineAttribute<float>("Geometry/theta_scale", geo_->theta_scale);
	io.DefineAttribute<float>("Geometry/q", geo_->qsf);
	io.DefineAttribute<float>("Geometry/zero_center", geo_->zeta_center);
	io.DefineAttribute<float>("Geometry/jacobian", geo_->jacobian_h, grids_->Nz);
	io.DefineAttribute<float>("Geometry/grho", geo_->grho_h, grids_->Nz);
}

void Adios::SetShapeFieldsKREHM()
{
  // no decomposition for fields data
  shape[0] = static_cast<unsigned int>(grids_->Ny);
  shape[1] = static_cast<unsigned int>(grids_->Nx);
  shape[2] = static_cast<unsigned int>(grids_->Nz);

  count[0] = static_cast<unsigned int>(grids_->Ny);
  count[1] = static_cast<unsigned int>(grids_->Nx);
  count[2] = static_cast<unsigned int>(grids_->Nz);
}

int Adios::AddSpectraGKVar(std::string name, int ndims, size_t sh[], size_t c[], size_t st[])
{
	// decomposition on the species and hermite
	if (ndims == 1) return -1;
	adios2::Dims varShape(ndims - 1);
	adios2::Dims varCount(ndims - 1);
	adios2::Dims varStart(ndims - 1);
	// skip the first dimension (time)
	for (int i=1; i<ndims; i++)
	{
		// if shape is 0, there is no decomposition on this dimension
		if (sh[i] == 0) varShape[i-1] = static_cast<unsigned int>(c[i]);
		else varShape[i-1] = static_cast<unsigned int>(sh[i]);
		varCount[i-1] = static_cast<unsigned int>(c[i]);
		varStart[i-1] = static_cast<unsigned int>(st[i]);
	}
	spectraVars.push_back(io.DefineVariable<float>(name, varShape, varStart, varCount));
	return spectraVars.size() - 1;
}

Adios::~Adios()
{
  writer.Close();
} 
