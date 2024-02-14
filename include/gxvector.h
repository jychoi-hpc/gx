#ifndef GXVECTOR_H
#define GXVECTOR_H

#include "moments.h"

#include <sundials/sundials_nvector.h>

/*
	Header file for the GXVector class used in the associated SUNDIALS NVector API

	GXVector is a thin wrapper around a std::vector<MomentsG> that has nSpecies entries.
	GXVector can be treated as an array of MomentsG by every other class in GX.
 */

class GXVector {
	public:
		using VecType = std::vector<MomentsG>;
		using SizeType = VecType::size_type;

		MomentsG& operator[]( SizeType i ) { return array[ i ]; };
		MomentsG const & operator[]( SizeType i ) const { return array[ i ]; };

		GXVector( Parameters* , Grids*, SUNContext ); // Construct underlying MomentsG from settings

		explicit GXVector( GXVector const& ); // Cloning constructor. THIS IS NOT A COPY CONSTRUCTOR (hence the 'explicit')

		static N_Vector CreateNVector( Parameters*, Grids*, SUNContext * );

		void setZero();

		void LinearSum( float, GXVector const&, float, GXVector const& ); // Sets the current object to be a*v_1 + b*v_2
		void SetConst( float );

		void Div( GXVector const&, GXVector const& );
		void Prod( GXVector const&, GXVector const& );

		void SetScaled( float, GXVector const& );
		void Scale( float );
		void SetAbs( GXVector const& );
		void SetInv( GXVector const& );

		GXVector & operator+=( GXVector const & );
		GXVector & operator=( GXVector const & );

		float MaxNorm() const;
		float WrmsNorm( GXVector const & ) const;


		float MinReal() const;

	private:
		SUNContext ctx;
		VecType array;
};



#endif // GXVECTOR_H
