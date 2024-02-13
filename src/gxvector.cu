
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

__global__ void set_constant_kernel( cuComplex* res, float x )
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();

  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      res[ig] = x;
  }
}

void GXVector::setConst( float c )
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

void GXVector::SetScaled( float c, GXVector const & other )
{
	*this = other;
	this->Scale( c );
}

void GXVector::Scale( float c )
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

float GXVector::MaxNorm()
{
	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(array[ 0 ].grids_, modes, modesRed);

	// allocate g-sized temporary object
	GXVector tmp( *this );

	// tmp[i] = ||this[i]||
	tmp.SetAbs( *this );

	// Allocate space for answer
	float *MaxElement;
	checkCuda(cudaMalloc(&MaxElement,  sizeof(float)));
	cudaMemset(MaxElement, 0., sizeof(float));

	// Take max over elements of tmp
	reducer.Max( tmp.array[ 0 ].G(), MaxElement );
	float cpuMaxElem;
	CP_TO_CPU( &cpuMaxElem, MaxElement );

	// Clean up
	cudaFree( &MaxElement );

	return cpuMaxElem;
}

// Set res_i = w_i * |g_i|^2
__global__ void wrmsKernel(cuComplex* res, cuComplex* g, cuComplex* w)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
	 // We know w is real, so w[ ig ].y == 0
	 res[ ig ].x = w[ ig ].x * cuCabsf( g[ ig ] );
	 res[ ig ].y = 0.0f;
  }
}

float GXVector::WrmsNorm( GXVector const & w )
{
	assert( w.array.size() == array.size() );
	// allocate g-sized temporary object
	GXVector tmp( *this );

	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(tmp.array[ 0 ].grids_, modes, modesRed);

	// Allocate space on GPU for answer
	float *SumResult;
	checkCuda(cudaMalloc(&SumResult,  sizeof(float)));
	cudaMemset(SumResult, 0., sizeof(float));


	// do tmp_i = w_i||g_i||^2 on GPU
	for( int i = 0; i < array.size(); ++i )
	{
		wrmsKernel<<< m.dG_all, m.dB_all >>> ( tmp.array[ i ].G(), array[ i ].G(), w.array[ i ].G() );
	}

	// tmp now contains {w_i ||g_i||^2 }
	// Sum all of tmp to get the answer
	
	reducer.Sum( tmp.array[ 0 ].G(), SumResult );
	float cpuSumResult;
	CP_TO_CPU( &cpuSumResult, SumResult );

	// Clean up
	cudaFree( &SumResult );

	return cpuSumResult;
}

__global__ void minRealKernel(cuComplex* res, cuComplex* in)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
	 res[ ig ].x = -in[ ig ].x;
	 res[ ig ].y = 0.0f;
  }
}

// Return minimum real part of all elements of g
float GXVector::MinReal()
{
	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(array[ 0 ].grids_, modes, modesRed);

	// allocate g-sized temporary object
	GXVector tmp( *this );

	// Allocate space for answer
	float *MaxElement;
	checkCuda(cudaMalloc(&MaxElement,  sizeof(float)));
	cudaMemset(MaxElement, 0., sizeof(float));

	// do tmp_i = -this[i] on GPU
	for( int i = 0; i < array.size(); ++i )
	{
		wrmsKernel<<< m.dG_all, m.dB_all >>> ( tmp.array[ i ].G(), array[ i ].G(), w.array[ i ].G() );
	}

	// tmp now contains -this
	// so max of tmp is min of *this
	reducer.Max( tmp.array[ 0 ].G(), MaxElement );
	float cpuMaxElem;
	CP_TO_CPU( &cpuMaxElem, MaxElement );

	// Clean up
	cudaFree( &MaxElement );

	// flip sign once again
	return -cpuMaxElem;
}
