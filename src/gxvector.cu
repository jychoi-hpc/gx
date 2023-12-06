
#include "gxvector.h"
#include <cuComplex.h>

/*
	This file provides the GXVector class, which is the building block of the custom NVector implementation
	for interfacing with SUNDIALS ARKODE.

	Author: Ian Abel (2023)
*/

GXVector::GXVector( Parameters *p, Grids *g )
{
	for( int i = 0; i < g->Nspecies; ++i )
	{
		int i_global = i + g->is_lo; // add local index offset
		array.emplace_back( p, g, i_global ); // construct MomentsG in place
	}
}

GXVector::GXVector( GXVector const& other )
{
	for( int i = 0; i < other.array.size(); ++i )
	{
		MomentsG& element = other[i];
		// This creates a new MomentsG, so allocates a new G_lm with the same parameters / grids / species indices
		// which is what the 'Clone' NVector op requires -- new memory, uninitialised, same everything else
		array.emplace_back( element.pars_, element.grids_, element.is_glob_ );
	}
}

void GXVector::setZero() 
{
	for( auto &m : array )
	{
		m.set_zero();
	}
}

__global__ void set_constant_kernel( cuComplex* res, double x )
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();

  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      res[ig] = x;
  }
}

void GXVector::setConst( double c )
{
	for( auto &m : array )
	{
		set_constant_kernel <<< m.dG_all, m.dB_all >>> ( m.G(), c );
	}
}

GXVector & GXVector::operator=( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	for( int i = 0; i < array.size(); ++i )
		array[ i ].copyFrom( &other.array[ i ] );
	return *this;
}

void GXVector::SetScaled( double c, GXVector const & other )
{
	*this = other;
	this->Scale( c );
}

void GXVector::Scale( double c )
{
	for( auto &m : array )
		m.scale( c );
}

__global__ void set_inv_kernel(cuComplex* res, cuComplex* in)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
    res[ig] = 1.0 / in[ig];
  }
}

void GXVector::SetInv( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	for( int i = 0; i < array.size(); ++i )
	{
		set_inv_kernel <<< m.dG_all, m.dB_all >>> ( array[ i ].G(), other.array[ i ].G() );
	}
}

__global__ void set_abs_kernel(cuComplex* res, cuComplex* in)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
	 res[ ig ].x = cuCabsf( in[ ig ] );
	 res[ ig ].y = 0.0f;
  }
}

void GXVector::SetAbs( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	for( int i = 0; i < array.size(); ++i )
	{
		set_abs_kernel <<< m.dG_all, m.dB_all >>> ( array[ i ].G(), other.array[ i ].G() );
	}
}

