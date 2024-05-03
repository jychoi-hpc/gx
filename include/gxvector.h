#ifndef GXVECTOR_H
#define GXVECTOR_H

#include "moments.h"

#include <sundials/sundials_core.h>
#include <sundials/sundials_nvector.h>
extern "C" {
#include <stdio.h>
}

/*
	Header file for the GXVector class used in the associated SUNDIALS NVector API

	GXVector is a thin wrapper around a std::vector<MomentsG> that has nSpecies entries.
	GXVector can be treated as an array of MomentsG by every other class in GX.
 */

class GXVector {
	public:
		using VecType = std::vector<MomentsG*>;
		using SizeType = VecType::size_type;

		MomentsG* operator[]( SizeType i ) { return array[ i ]; };
		MomentsG const * operator[]( SizeType i ) const { return array[ i ]; };

		virtual ~GXVector() { 
			if( owns_data ) {
				for( auto m_ptr : array ) {
					delete m_ptr;
				}
			}
		}

		GXVector( Parameters* , Grids*, SUNContext ); // Construct underlying MomentsG from settings

		explicit GXVector( GXVector const& ); // Cloning constructor. THIS IS NOT A COPY CONSTRUCTOR (hence the 'explicit')

		operator MomentsG** () { return array.data(); };

		static N_Vector CreateNVector( Parameters*, Grids*, SUNContext );
		static N_Vector CreateEmptyNVector( SUNContext );

		N_Vector asNVector();

		void AddConst( sunrealtype ); // this += b


		void SetZero();

		void LinearSum( float, GXVector const&, float, GXVector const& ); // Sets the current object to be a*v_1 + b*v_2
		void SetConst( float );

		// Elementwise division ; this[i] = a[i]/b[i]
		void Div( GXVector const&, GXVector const& );
		// Elementwise product  ; this[i] = a[i]*b[i]
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

		explicit GXVector( MomentsG **G, SUNContext ctx_ ) :
			ctx(ctx_),owns_data(false)
		{ 
			array.resize( G[0]->grids_->Nspecies );
			for( int i = 0; i < array.size(); ++i )
				array[ i ] = G[ i ];
		};

		void sync();
		void update_tprim( double );

		SizeType nSpecies() const { return array.size(); } ;

		SUNContext ctx;
	private:
		bool owns_data;
		VecType array;
};



#endif // GXVECTOR_H
