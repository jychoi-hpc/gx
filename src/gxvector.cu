
#include "gxvector.h"

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


