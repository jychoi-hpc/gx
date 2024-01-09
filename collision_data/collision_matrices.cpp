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
  if (j<0) return 0.0;
  else return 1./pow(sqrt(2.),j)/sqrt(factorial(j))*std::hermite(j,x/sqrt(2.));
}

double laguerreProj(int k, double x) {
  return pow(-1,k)*std::laguerre(k, x);
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

double nuPar(double ua, double ub) {
  return 2./pow(ua, 3)*(-1/sqrt(M_PI)/ub*exp(-ub*ub) + erf(ub)/(2*ub*ub));
}

double du2nuPar(double u) {
  return 6*exp(-u*u)/sqrt(M_PI)/pow(u,3) + 4*exp(-u*u)/sqrt(M_PI)/u - 3*erf(u)/pow(u,4);
}

double nuE(double u) {
  return -8./sqrt(M_PI)/u/u*exp(-u*u) + 2./pow(u,3)*erf(u);
}

double nuS(double u) {
  return -4./sqrt(M_PI)/u/u*exp(-u*u) + 2./pow(u,3)*erf(u);
}

double Dnu(double u) {
  return nuD(u,u) - nuS(u);
}

extern "C"
double nuDLorentz_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double vp = xi*u*sqrt(2);
  double mB = u*u*(1-xi*xi);

  double val = hermiteProj(k, vp)*laguerreProj(j, mB)
    *nuD(u,u)*u*u*sqrt(8)
    *(
      l*sqrt((1 + m)*(2 + m))*hermiteExpd(2 + m, vp)*laguerreExpd(-1 + l, mB) - (l + m + 2*l*m)*hermiteExpd(m, vp)*laguerreExpd(l, mB) + 
      (1 + l)*sqrt((-1 + m)*(m))*hermiteExpd(-2 + m, vp)*laguerreExpd(1 + l, mB)
    );
  return val;
}

extern "C"
double invU3Lorentz_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double vp = xi*u*sqrt(2);
  double mB = u*u*(1-xi*xi);

  double val = hermiteProj(k, vp)*laguerreProj(j, mB)
    *1/u*sqrt(8)
    *(
      l*sqrt((1 + m)*(2 + m))*hermiteExpd(2 + m, vp)*laguerreExpd(-1 + l, mB) - (l + m + 2*l*m)*hermiteExpd(m, vp)*laguerreExpd(l, mB) + 
      (1 + l)*sqrt((-1 + m)*(m))*hermiteExpd(-2 + m, vp)*laguerreExpd(1 + l, mB)
    );
  return val;
}

extern "C"
double nuParDiffT0_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double vp = xi*u*sqrt(2);
  double mB = u*u*(1-xi*xi);

  double val = hermiteProj(k, vp)*laguerreProj(j, mB)
    *u*u*sqrt(8)
    *(
     nuPar(u,u)*(
      (-(sqrt((-1 + m)*m)*m*hermiteExpd(-2 + m, vp)*laguerreExpd(l, mB)) - 
      sqrt((1 + m)*(2 + m))*hermiteExpd(2 + m, vp)*(2*l*laguerreExpd(-1 + l, mB) + 
      (2*l + m)*laguerreExpd(l, mB)) - 2*sqrt((-1 + m)*m)*hermiteExpd(-2 + m, vp)*
      (l*laguerreExpd(l, mB) + (1 + l)*laguerreExpd(1 + l, mB)) + 
      hermiteExpd(m, vp)*(-2*l*(2*l + m)*laguerreExpd(-1 + l, mB) - 
      (4*l + 8*l*l + m + 4*l*m + 2*m*m)*laguerreExpd(l, mB) - 
      2*(1 + l)*(2*l + m)*laguerreExpd(1 + l, mB)))/2
     ) + du2nuPar(u)/u*(
      (sqrt((-1 + m)*m)*hermiteExpd(-2 + m, vp)*laguerreExpd(l, mB) + 
      hermiteExpd(m, vp)*(2*l*laguerreExpd(-1 + l, mB) + (2*l + m)*laguerreExpd(l, mB)))/2
     )
    );
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
    *u*u*sqrt(8)
    *u*u/2*nuD(u,u)*(1+xi*xi);
  return val;
}

// projection of nuPar*u^2/2*(1-xi^2)*h
// this is term proportional to b = kperp^2 rho^2
extern "C"
double nuParFLR_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *u*u*sqrt(8)
    *u*u/2*(nuPar(u,u)*(1-xi*xi));
  return val;
}

// projection of nuD*u^2/2*(1+xi^2)*h + nuPar*u^2/2*(1-xi^2)*h
// this is term proportional to b = kperp^2 rho^2
extern "C"
double nuFLR_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *u*u*sqrt(8)
    *u*u/2*(nuD(u,u)*(1+xi*xi) + nuPar(u,u)*(1-xi*xi));
  return val;
}

// projection of (1/u^3)*u^2/2*(1+xi^2)*h
// this is term proportional to b = kperp^2 rho^2
extern "C"
double invU3FLR_proj(int n, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int j = ((int*)user_data)[0];
  int k = ((int*)user_data)[1];
  int l = ((int*)user_data)[2];
  int m = ((int*)user_data)[3];

  double val = hermiteProj(k, xi*u*sqrt(2))*laguerreProj(j,u*u*(1-xi*xi)) 
    *hermiteExpd(m,xi*u*sqrt(2))*laguerreExpd(l,u*u*(1-xi*xi)) 
    *u*u*sqrt(8)
    *0.5*(1/u)*(1+xi*xi);
  return val;
}

extern "C"
double alphaE_proj(int n_, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int l = ((int*)user_data)[0];
  int m = ((int*)user_data)[1];
  int n = ((int*)user_data)[2];

  double val = laguerreProj(n, u*u*(1-xi*xi))
    *laguerreExpd(l, u*u*(1-xi*xi))*hermiteExpd(m, xi*u*sqrt(2))
    *u*u*sqrt(8)
    *nuE(u)*u*u;
}

extern "C"
double alphaParL_proj(int n_, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int l = ((int*)user_data)[0];
  int m = ((int*)user_data)[1];
  int n = ((int*)user_data)[2];

  double val = laguerreProj(n, u*u*(1-xi*xi))
    *laguerreExpd(l, u*u*(1-xi*xi))*hermiteExpd(m, xi*u*sqrt(2))
    *u*u*sqrt(8)
    *nuD(u,u)*xi*u*sqrt(2);
}

extern "C"
double alphaParD_proj(int n_, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int l = ((int*)user_data)[0];
  int m = ((int*)user_data)[1];
  int n = ((int*)user_data)[2];

  double val = laguerreProj(n, u*u*(1-xi*xi))
    *laguerreExpd(l, u*u*(1-xi*xi))*hermiteExpd(m, xi*u*sqrt(2))
    *u*u*sqrt(8)
    *Dnu(u)*xi*u*sqrt(2);
}

extern "C"
double alphaPerpL_proj(int n_, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int l = ((int*)user_data)[0];
  int m = ((int*)user_data)[1];
  int n = ((int*)user_data)[2];

  double val = (laguerreProj(n, u*u*(1-xi*xi)) + laguerreProj(n+1, u*u*(1-xi*xi)))
    *laguerreExpd(l, u*u*(1-xi*xi))*hermiteExpd(m, xi*u*sqrt(2))
    *u*u*sqrt(8)
    *nuD(u,u);
}

extern "C"
double alphaPerpD_proj(int n_, double *x, void *user_data) {
  double u = x[0];
  double xi = x[1];

  int l = ((int*)user_data)[0];
  int m = ((int*)user_data)[1];
  int n = ((int*)user_data)[2];

  double val = (laguerreProj(n, u*u*(1-xi*xi)) + laguerreProj(n+1, u*u*(1-xi*xi)))
    *laguerreExpd(l, u*u*(1-xi*xi))*hermiteExpd(m, xi*u*sqrt(2))
    *u*u*sqrt(8)
    *Dnu(u);
}
