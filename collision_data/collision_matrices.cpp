#include <math.h>
#include <stdio.h>
#include <cmath>
using namespace std;

double factorial(int m) {
  if (m <2) return 1.;
  if (m==2) return 2.;
  if (m==3) return 6.; 
  if (m==4) return 24.;
  if (m==5) return 120.;
  if (m==6) return 720.;
  else return m*factorial(m-1);
} 

double hermiteProj(int j, double x) {
  return 1./pow(sqrt(2.),j)/sqrt(factorial(j))*hermite(j,x/sqrt(2.));
}

double laguerreProj(int k, double x) {
  return pow(-1,k)*laguerre(k, x);
}

double hermiteExpd(int j, double x) {
  return hermiteProj(j, x)*exp(-x*x/2.)/sqrt(2.*M_PI);
}

double laguerreExpd(int j, double x) {
  return laguerreProj(j, x)*exp(-x);
}

double nuD(double ua, double ub) {
  return 1./pow(ua, 3)*(1/sqrt(M_PI)/ub*exp(-ub*ub) + erf(ub)*(1-1/(2*ub*ub)));
}

extern "C"
double nuD_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *nuD(u,u)*u*u*sqrt(8);
  return val;
}

extern "C"
double invU3_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *1/u*sqrt(8);
  return val;
}

// projection of nuD*u^2/2*(1+xi^2)*h
// this is term proportional to b = kperp^2 rho^2
extern "C"
double nuDFLR_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *nuD(u,u)*u*u*sqrt(8)
    *u*u/2*(1+xi*xi);
  return val;
}
