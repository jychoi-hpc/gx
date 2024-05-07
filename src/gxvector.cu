
#include "gxvector.h"
#include "device_funcs.h"
#include "reductions.h"

extern "C" {
	#include <stdio.h>
}


/*
	This file provides the GXVector class, which is the building block of the custom NVector implementation
	for interfacing with SUNDIALS ARKODE.

	Author: Ian Abel (2023)
*/

GXVector::GXVector( Parameters *p, Grids *g, SUNContext Context )
	: ctx( Context ), owns_data( true )
{
	for( int i = 0; i < g->Nspecies; ++i )
	{
		int i_global = i + g->is_lo; // add local index offset
		array.emplace_back( new MomentsG( p, g, i_global ) ); // construct MomentsG in place
	}
}

GXVector::GXVector( GXVector const& other )
	: owns_data( true )
{
	ctx = other.ctx;
	for( int i = 0; i < other.array.size(); ++i )
	{
		MomentsG const& element = *(other[i]);
		// This creates a new MomentsG, so allocates a new G_lm with the same parameters / grids / species indices
		// which is what the 'Clone' NVector op requires -- new memory, uninitialised, same everything else
		array.emplace_back( new MomentsG( element.pars_, element.grids_, element.is_glob_ ) );
	}
}

void GXVector::SetZero() 
{
	for( auto &m : array )
	{
		m->set_zero();
	}
}

void GXVector::SetConst( float c )
{
	for( auto &m : array )
	{
		set_constant_kernel <<< m->dG_all, m->dB_all >>> ( *m, c );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

GXVector & GXVector::operator=( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	for( int i = 0; i < array.size(); ++i )
		array[ i ]->copyFrom( other.array[ i ] );
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
		m->scale( c );
}

void GXVector::SetInv( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	MomentsG& m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		set_inv_kernel <<< m.dG_all, m.dB_all >>> ( *(array[ i ]), *(other.array[ i ]) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

void GXVector::SetAbs( GXVector const & other )
{
	assert( other.array.size() == array.size() );
	MomentsG& m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		set_abs_kernel <<< m.dG_all, m.dB_all >>> ( *(array[ i ]), *(other.array[ i ]) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

float GXVector::MaxNorm() const
{
	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(array[ 0 ]->grids_, modes, modesRed);

	// allocate temporary object for |g_i|
	float *tmp;
	
	// We need same number of floats as complex numbers in g
	size_t size_per_species = array[0]->getN() * sizeof(float);
	checkCuda(cudaMalloc((void**) &tmp, size_per_species * array.size() )); 

	// Allocate space on GPU for answer
	float *MaxElement;
	checkCuda(cudaMalloc(&MaxElement,  sizeof(float)));
	cudaMemset(MaxElement, 0., sizeof(float));

	// do tmp_i = ||g_i|| on GPU
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		float *tmp_species_i = tmp + i * array[0]->getN();
		absValKernel<<< m.dG_all, m.dB_all >>> ( tmp_species_i, *(array[ i ]) );
	}

	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );

	// Take max over elements of tmp
	reducer.Max( tmp, MaxElement );
	float cpuMaxElem;
	
	checkCuda( cudaDeviceSynchronize() );
	CP_TO_CPU( &cpuMaxElem, MaxElement, sizeof(float) );

	// Clean up
	cudaFree( &MaxElement );
	cudaFree( &tmp );

	return cpuMaxElem;
}

float GXVector::WrmsNorm( GXVector const & w ) const
{
	assert( w.array.size() == array.size() );
	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(array[ 0 ]->grids_, modes, modesRed);

	// allocate temporary object for w_i |g_i|^2
	float *tmp;
	
	// We need same number of floats as complex numbers in g
	size_t size_per_species = array[0]->getN() * sizeof(float);
	checkCuda(cudaMalloc((void**) &tmp, size_per_species * array.size() )); 

	// Allocate space on GPU for answer
	float *SumResult;
	checkCuda(cudaMalloc(&SumResult,  sizeof(float)));
	cudaMemset(SumResult, 0., sizeof(float));


	// do tmp_i = w_i^2 ||g_i||^2 on GPU
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		float *tmp_species_i = tmp + i * array[0]->getN();
		wrmsKernel<<< m.dG_all, m.dB_all >>> ( tmp_species_i, *(array[ i ]), *(w.array[ i ]) );
	}

	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );

	// tmp now contains {w_i^2 ||g_i||^2 }
	// Sum all of tmp to get the answer
	
	reducer.Sum( tmp, SumResult );
	float cpuSumResult;
	checkCuda( cudaDeviceSynchronize() );
	CP_TO_CPU( &cpuSumResult, SumResult, sizeof(float) );

	// Clean up
	cudaFree( &SumResult );
	cudaFree( &tmp );

	// The norm we want is sqrt( Sum (w_i^2 |g_i|^2) / n )
	// where n is the number of actual degrees of freedom in g
	// so nspecies * # DoF in a MomentsG
	size_t TotalDegreesOfFreedom = array.size() * array[0]->getDegreesOfFreedom();
	return sqrtf(cpuSumResult/static_cast<float>(TotalDegreesOfFreedom));
}

// Return minimum real part of all elements of g
float GXVector::MinReal() const
{
	// Reduction object
	std::vector<int32_t> modes{'y', 'x', 'z', 'l', 'm', 's'};
	std::vector<int32_t> modesRed{};
	Reduction<float> reducer(array[ 0 ]->grids_, modes, modesRed);

	// allocate temporary object for real components
	float *tmp;

	// We need same number of floats as complex numbers in g
	size_t size_per_species = array[0]->getN() * sizeof(float);
	checkCuda(cudaMalloc((void**) &tmp, size_per_species * array.size() )); 

	// Allocate space for answer
	float *MaxElement;
	checkCuda(cudaMalloc(&MaxElement,  sizeof(float)));
	cudaMemset(MaxElement, 0., sizeof(float));

	// do tmp_i = - Re( this[i] ) on GPU
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		float *tmp_species_i = tmp + i * array[0]->getN();
		minusRealKernel<<< m.dG_all, m.dB_all >>> ( tmp_species_i, *(array[ i ]) );
		checkCuda( cudaGetLastError() );
	}

	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );

	// tmp now contains -this
	// so max of tmp is min of *this
	reducer.Max( tmp, MaxElement );
	float cpuMaxElem;
	checkCuda( cudaDeviceSynchronize() );
	CP_TO_CPU( &cpuMaxElem, MaxElement, sizeof(float) );

	// Clean up
	cudaFree( &MaxElement );
	cudaFree( &tmp );

	// flip sign -- we want minimum not maximum element
	return -cpuMaxElem;
}

void GXVector::Div( GXVector const& x, GXVector const& y )
{
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		elem_div_kernel<<< m.dG_all, m.dB_all >>> ( *(array[ i ]), *(x.array[ i ]), *(y.array[ i ]) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

void GXVector::Prod( GXVector const& x, GXVector const& y )
{
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		elem_prod_kernel<<< m.dG_all, m.dB_all >>> ( *(array[ i ]), *(x.array[ i ]), *(y.array[ i ]) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

// Sets the current object to be a*x + b*y
void GXVector::LinearSum( float a, GXVector const& x, float b, GXVector const& y )
{
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		// Note the last argument is true to force-ignore any eqfix nonsense
		add_scaled_kernel<<< m.dG_all, m.dB_all >>> ( *(array[ i ]), a, *(x.array[ i ]), b, *(y.array[ i ]), true );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}

GXVector & GXVector::operator+=( GXVector const& other )
{
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		accumulate_kernel<<< m.dG_all, m.dB_all >>> ( *(array[ i ]), *(other.array[ i ]) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
	return *this;
}

void GXVector::sync()
{
	int Nspecies = array[ 0 ]->grids_->Nspecies;
	for( int is = 0; is < Nspecies; ++is ) {
		array[ is ]->sync();
	}
}

void GXVector::update_tprim( double t )
{
	int Nspecies = array[ 0 ]->grids_->Nspecies;
	for( int is = 0; is < Nspecies; ++is ) {
		array[ is ]->update_tprim( t );
	}
}

void GXVector::AddConst( sunrealtype b )
{
	MomentsG const & m = *(array[ 0 ]);
	for( int i = 0; i < array.size(); ++i )
	{
		add_const_kernel<<< m.dG_all, m.dB_all >>> ( *(array[ i ]), static_cast<float>(b) );
		checkCuda( cudaGetLastError() );
	}
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaGetLastError() );
}
