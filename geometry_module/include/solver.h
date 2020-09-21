#pragma once
#include "geometric_coefficients.h"
#include "vmec_variables.h"

template<class T>void PolyFit(//<=========FITS A POLYNOMIAL TO A SET OF POINTS
 const T*X,//<-------INDEPENDENT-VARIABLE VALUES (EACH X[i] MUST BE UNIQUE)
 const T*Y,//<-------------------DEPENDENT-VARIABLE VALUES (SAME SIZE AS X)
 int n,//<-------------------------------------NUMBER OF ELEMENTS IN X OR Y
 int m,//<----------------NUMBER OF COEFFICIENTS IN THE BEST-FIT POLYNOMIAL
 T*C){//<------STORAGE FOR COEFFICIENTS (SIZE=m) y=C[0]+C[1]*x+C[2]*x^2+...
 T**A=new T*[m];/*<-*/for(int i=0,j,k;i<m;++i){//............augmented matrix
 for(A[i]=new T[m+1],j=0;j<m+1;++j)A[i][j]=0;
 for(k=0;k<n;++k)for(A[i][m]+=pow(X[k],i)*Y[k],j=0;j<m;++j)
 A[i][j]+=pow(X[k],i+j);}
 for(int i=1,j,k;i<m;++i)for(k=0;k<i;++k)for(j=m;j>=0;--j)//.........Gaussian
 A[i][j]-=A[i][k]*A[k][j]/A[k][k];// elimination
 for(int i=m-1,j;i>=0;--i)for(C[i]=A[i][m]/A[i][i],j=i+1;j<m;++j)//..backward
 C[i]-=C[j]*A[i][j]/A[i][i];// substitution
 for(int i=0;i<m;++i)delete[]A[i];/*&*/delete[]A;
}

struct g_params {
  double theta_pest_target_f;
  double zeta0_f;
  int vmec_radial_index_half_f[2];
  double vmec_radial_weight_half_f[2];
  VMEC_variables *vmec_f;
};

double fzero_residual(double, void*);
void solver_vmec_theta(double*, double*, int, double, double, VMEC_variables*, int*, double*);
void interp_to_new_grid(double*, double*, double*, int, bool);
double find_zero_crossing(double*, double*, int);
