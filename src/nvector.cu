
#include "gxvector.h"

/*
	API backend to wrap the memebr functions of GXVector into SUNDIAL NVector
 */

realtype GXV_MinReal( N_Vector z );
realtype GXV_WrmsNorm( N_Vector z, N_Vector w );
realtype GXV_MaxNorm( N_Vector z );
realtype GXV_MaxNorm( N_Vector z );
void GXV_AddConst( N_Vector x, realtype b, N_Vector out );
void GXV_Inv( N_Vector in, N_Vector out );
void GXV_Abs( N_Vector in, N_Vector out );
void GXV_Scale( realtype c, N_Vector x, N_Vector z );
void GXV_Prod( N_Vector x, N_Vector y, N_Vector z );
void GXV_Div( N_Vector x, N_Vector y, N_Vector z );
void GXV_Const( realtype c, N_Vector z );
void GXV_LinearSum( realtype a, N_Vector v, realtype b, N_Vector w, N_Vector out );
N_Vector GXV_Clone( N_Vector other );
void GXV_Destroy( N_Vector v );
N_Vector_ID GXV_GetVectorID( N_Vector );

struct _generic_N_Vector_Ops GXVOps = {
   .nvgetvectorid = GXV_GetVectorID,
   .nvclone = GXV_Clone,
   .nvcloneempty = nullptr,
   .nvdestroy = GXV_Destroy,
   .nvspace = nullptr,
   .nvgetarraypointer = nullptr,
   .nvgetdevicearraypointer = nullptr,
   .nvsetarraypointer = nullptr,
   .nvgetcommunicator = nullptr,
   .nvgetlength = nullptr,
   .nvgetlocallength = nullptr,
   .nvlinearsum = GXV_LinearSum,
   .nvconst = GXV_Const,
   .nvprod = GXV_Prod,
   .nvdiv = GXV_Div,
   .nvscale = GXV_Scale,
   .nvabs = GXV_Abs,
   .nvinv = GXV_Inv,
   .nvaddconst = GXV_AddConst,
   .nvdotprod = nullptr,
   .nvmaxnorm = GXV_MaxNorm,
   .nvwrmsnorm = GXV_WrmsNorm,
   .nvwrmsnormmask = nullptr,
   .nvmin = GXV_MinReal,
   .nvwl2norm = nullptr,
   .nvl1norm = nullptr,
   .nvcompare = nullptr,
   .nvinvtest = nullptr,
   .nvconstrmask = nullptr,
   .nvminquotient = nullptr,
   .nvlinearcombination = nullptr,
   .nvscaleaddmulti = nullptr,
   .nvdotprodmulti = nullptr,
   .nvlinearsumvectorarray = nullptr,
   .nvscalevectorarray = nullptr,
   .nvconstvectorarray = nullptr,
   .nvwrmsnormvectorarray = nullptr,
   .nvwrmsnormmaskvectorarray = nullptr,
   .nvscaleaddmultivectorarray = nullptr,
   .nvlinearcombinationvectorarray = nullptr,
   .nvdotprodlocal = nullptr,
   .nvmaxnormlocal = nullptr,
   .nvminlocal = nullptr,
   .nvl1normlocal = nullptr,
   .nvinvtestlocal = nullptr,
   .nvconstrmasklocal = nullptr,
   .nvminquotientlocal = nullptr,
   .nvwsqrsumlocal = nullptr,
   .nvwsqrsummasklocal = nullptr,
   .nvdotprodmultilocal = nullptr,
   .nvdotprodmultiallreduce = nullptr,
   .nvbufsize = nullptr,
   .nvbufpack = nullptr,
   .nvbufunpack = nullptr,
};

N_Vector GXVector::CreateNVector( Parameters *pars, Grids *grids, SUNContext ctx )
{
	N_Vector z = N_VNewEmpty( ctx );
	z->ops = &GXVOps;
	z->content = new GXVector( pars, grids, ctx );
	return z;
}

N_Vector GXVector::asNVector();
{
	N_Vector z = N_VNewEmpty( ctx );
	z->ops = &GXVOps;
	z->content = this;
	return z;
}

#define GXV( nv ) reinterpret_cast<GXVector*>( nv->content )

N_Vector_ID GXV_GetVectorID( N_Vector )
{
	return SUNDIALS_NVEC_CUSTOM;
}

N_Vector GXV_Clone( N_Vector other )
{
	N_Vector new_vector = N_VNewEmpty( GXV( other )->ctx );
	new_vector->ops = &GXVOps;
	new_vector->content = new GXVector( *GXV( other ) );
	return new_vector;
}

void GXV_Destroy( N_Vector v )
{
	delete v->content;
	v->content = nullptr;
	v->ops = nullptr;
	N_VFreeEmpty( v );
}

void GXV_LinearSum( realtype a, N_Vector v, realtype b, N_Vector w, N_Vector out )
{
	GXV( out )->LinearSum( a, *GXV( v ), b, *GXV( w ) );
}

void GXV_Const( realtype c, N_Vector z )
{
	GXV( z )->SetConst( c );
}

void GXV_Div( N_Vector x, N_Vector y, N_Vector z )
{
	GXV( z )->Div( *GXV( x ), *GXV( y ) );
}

void GXV_Prod( N_Vector x, N_Vector y, N_Vector z )
{
	GXV( z )->Prod( *GXV( x ), *GXV( y ) );
}

void GXV_Scale( realtype c, N_Vector x, N_Vector z )
{
	GXV( z )->SetScaled( c, *GXV( x ) );
}

void GXV_Abs( N_Vector in, N_Vector out )
{
	GXV( out )->SetAbs( *GXV( in ) );
}

void GXV_Inv( N_Vector in, N_Vector out )
{
	GXV( out )->SetInv( *GXV( in ) );
}

void GXV_AddConst( N_Vector x, realtype b, N_Vector out )
{
	GXVector & v = *GXV( out );
	GXVector & w = *GXV( x );

	v.SetConst( b );
	v += w;
}

realtype GXV_MaxNorm( N_Vector z )
{
	return GXV( z )->MaxNorm();
}

realtype GXV_WrmsNorm( N_Vector z, N_Vector w )
{
	return GXV( z )->WrmsNorm( *GXV( w ) );
}

realtype GXV_MinReal( N_Vector z )
{
	return GXV( z )->MinReal();
}


