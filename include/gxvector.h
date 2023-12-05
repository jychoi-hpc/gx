#ifndef GXVECTOR_H
#define GXVECTOR_H

#include "moments.h"

#include <sundials/sundials_nvector.h>

/*
	Header file for the GXVector class used in the associated SUNDIALS NVector API

	GXVector is a thin wrapper around a std::vecotr<MomentsG> that has nSpecies entries.
	GXVector can be treated as an array of MomentsG by every other class in GX.
 */

class GXVector {
	public:
		using VecType = std::vector<MomentsG>;
		using SizeType = VecType::size_type;

	MomentsG& operator[]( SizeType i ) { return array[ i ]; };

	GXVector( Parameters* , Grids* ); // Construct underlying MomentsG from settings

	explicit GXVector( GXVector const& ); // Cloning constructor. THIS IS NOT A COPY CONSTRUCTOR (hence the 'explicit')

	static N_Vector CreateNVector( Parameters*, Grids* );

	void LinearSum( double, GXVector const&, double, GXVector const& ); // Sets the current object to be a*v_1 + b*v_2
	void SetConst( double );

	void Div( GXVector const&, GXVector const& );
	void Prod( GXVector const&, GXVector const& );

	void SetScaled( double, GXVector const& );
	void SetAbs( GXVector const& );
	void SetInv( GXVector const& );

	GXVector & operator+=( GXVector const & );

	double MaxNorm() const;
	double WrmsNorm( GXVector const & ) const;


	double MinReal() const;

	private:
		VecType array;
		VecType::size_type nSpecies;
}



#endif // GXVECTOR_H
