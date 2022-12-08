__device__ unsigned int get_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

__device__ int iget_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ int iget_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ int iget_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

/*
__device__ unsigned int get_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ unsigned int get_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ unsigned int get_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}

__device__ int iget_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ int iget_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ int iget_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}
*/

__device__ float g0(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, err;

  if (b < tol) {return 1.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
  tk = __expf(-b);
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    tk  = tk * b2sq * xi * xi;
    g += tk;
    x  += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__device__ float g1(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, xp1i, err;

  if (b < tol) {return 0.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
  tk = __expf(-b) * b2;
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    xp1i=1./(1.+x);
    tk  = tk * b2sq * xi * xp1i;
    g += tk;
    x  += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__device__ float sgam0 (float b) {return sqrt(g0(b));}

__device__ float flr(float b) {
  float gam0 = g0(b);
  float gam1 = g1(b);
  float gam = sqrt(gam0);
  float rg = gam1/gam0;
  return -b/2. * (1.-rg) * gam;
}

__device__ float flr2(float b) {
  float gam0 = g0(b);
  float gam1 = g1(b);
  float gam = sqrt(gam0);
  float rg = gam1/gam0;
  return -(b/2.) * (1.+(1.-rg)*(1.-(b/2.)*(3.+rg))) * gam;
}

__device__ float nwgt(float b) {return 1./(1.+b/2.);}

__device__ float twgt(float b) {
  float d = 1./(1.+b/2.);
  return -b/2. * d * d;
}

__device__ float wgt(float b) {
  float gam0 = g0(b);
  return 1.-gam0;
}

__device__ float omegaD(float rho_vt, float kx, float ky, float shat, float gb,float gb0,float cv, float cv0)
{
  float shatInv;
  //if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  //} else {
  //  shatInv = 1.;
  //}

  
  return rho_vt* ( ky * (gb + cv) + kx * shatInv * (gb0 + cv0) );
  
}

__device__ double b(float rho, float kx, float ky, float shat, float gds2, float gds21, float gds22, float bmagInv)
{
  float shatInv;
  //if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  //} else {
  //  shatInv = 1.;
  //}

  double b = ( ky * (ky*gds2 + 2*kx*shatInv*gds21) + pow(kx*shatInv,2)*gds22 ) * pow(bmagInv,2) * pow(rho,2);
  return b;
}

__device__ float c(float kx, float gds22, float qsf, float eps, float bmagInv, float shat, float rho)
{ 
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  // -k_r rho_pol / B
  return kx*shatInv*sqrt(gds22)*qsf/eps*bmagInv*rho;
}

__host__ __device__ bool operator>(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y > g.x*g.x+g.y*g.y;
}

__host__ __device__ bool operator<(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y < g.x*g.x+g.y*g.y;
}


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) 
{
  return cuCaddf(f,g);
} 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
{
  return cuCsubf(f,g);
}  

__host__ __device__ cuComplex operator-(cuComplex f)
{
  cuComplex zero;
  zero.x = 0.;
  zero.y = 0.;
  return cuCsubf(zero,f);
}  
/*
__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}
*/
__host__ __device__ cuComplex operator*(double scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, double scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuComplex operator/(cuComplex f, float scaler)
{
  cuComplex result;
  result.x = f.x / scaler;
  result.y = f.y / scaler;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) 
{
  return cuCdivf(f,g);
}

__device__ int get_ikx(int idx) {
  if( idx<nx/2+1 )
    return idx;
  else
    return idx-nx;
}

