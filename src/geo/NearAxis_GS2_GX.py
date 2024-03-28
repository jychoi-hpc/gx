#!/opt/local/bin/python3

import os
import math
import numpy as np
from qsc.qsc import Qsc
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

## Point to the current location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pi = math.pi
mu0 = 4*pi*10**(-7)

## GS2 and GX input file parameters
nperiod  = 1
drhodpsi = 1.0
rmaj     = 1.0
kxfac    = 1.0

## Free parameters
etabar  = 0.9 # parametrization for all first order quasisymmetric fields
B0      = 1.0 # magnetic field strength on axis
raxis   = [1.0,0.045] # axis shape - radial coefficients
zaxis   = [0.0,-0.045] # axis shape - vertical coefficients
NFP     = 3 # number of field periods
phiEDGE = 0.5 # toroidal flux at the boundary divided by 2*pi

## Resolution
Nphi     = 250 # resolution along the axis
nlambda  = 10 # GS2 quantity: resolution in the pitch-angle variable
tgridmax = pi # maximum (and -1*minumum) of the field line coordinate
ntgrid   = 129 # resolution along the field line  

## Geometry and normalizations
alpha = 0.0 # field line label
shat  = 1.0 # used in the definition of GX and GS2 quantities
Aminor = 1.0 # minor radius - GS2 normalization parameter
normalizedtorFlux = 0.01 # normalization for the toroidal flux

## Output files
gs2gridNA = "gs2input.out"
gxgridNA  = "gx_near_axis.out"

#################################
############## RUN ##############
#################################

## Obtain near-axis configuration
stel = Qsc(rc=raxis,zs=zaxis, nfp=NFP, etabar=etabar, nphi=Nphi)
iota = abs(stel.iota)
sigmaSol = stel.sigma
Laxis = stel.axis_length
sprime = stel.d_l_d_phi
curvature = stel.curvature
phi = stel.phi
nNormal = stel.iotaN - stel.iota
varphi = stel.varphi

## Find near-axis sigma and sprime functions for this specific theta grid
sigmaTemp  = interp1d(phi,sigmaSol, kind='cubic')
sprimeTemp = interp1d(phi,sprime, kind='cubic')
curvTemp   = interp1d(phi,curvature, kind='cubic')

period=2*pi*(1-1/Nphi)/NFP
def phiToNFP(phi):
	if phi==0:
		phiP=0
	else:
		phiP=-phi%period
	return phiP

def sigma(phi):      return sigmaTemp(phiToNFP(phi))
def sprimeFunc(phi): return sprimeTemp(phiToNFP(phi))
def curvFunc(phi):   return curvTemp(phiToNFP(phi))

## GS2 geometric quantities
rVMEC 			        = -np.sqrt((2*phiEDGE*normalizedtorFlux)/B0)
def Phi(theta):         return (theta - alpha)/(iota - nNormal)
def bmagNew(theta):     return ((Aminor**2)*B0*(1+rVMEC*etabar*np.cos(theta)))/(2*phiEDGE)
def gradparNew(theta):  return  ((2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)/(sprimeFunc((alpha-theta)/(iota-nNormal))*2*pi/Laxis)
def gds2New(theta):     return (((Aminor**2)*B0)/(2*phiEDGE))*((etabar**2*np.cos(theta)**2)/curvFunc(Phi(theta))**2 + (curvFunc(Phi(theta))**2*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))**2)/etabar**2)
def gds21New(theta):    return -(1/(2*phiEDGE))*Aminor**2*shat*((B0*etabar**2*np.cos(theta)*np.sin(theta))/curvFunc(Phi(theta))**2+(1/etabar**2)*B0*(curvFunc(Phi(theta))**2)*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))*(-np.cos(theta)+np.sin(theta)*sigma(Phi(theta))))
def gds22New(theta):    return (Aminor**2*B0*(shat**2)*((etabar**4)*np.sin(theta)**2+(curvFunc(Phi(theta))**4)*(np.cos(theta)-np.sin(theta)*sigma(Phi(theta)))**2))/(2*phiEDGE*(etabar**2)*curvFunc(Phi(theta))**2)
def gbdriftNew(theta):  return (2*np.sqrt(2)*etabar*np.cos(theta))/np.sqrt(B0/phiEDGE)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdriftNew(theta):  return gbdriftNew(theta)
def gbdrift0New(theta): return -2*np.sqrt(2)*np.sqrt(phiEDGE/B0)*shat*etabar*np.sin(theta)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdrift0New(theta): return gbdrift0New(theta)
lambdamin=((2*phiEDGE)/((Aminor**2)*B0))/(1+abs(rVMEC*etabar))
lambdamax=((2*phiEDGE)/((Aminor**2)*B0))/(1-abs(rVMEC*etabar))
lambdavec=np.linspace(lambdamin,lambdamax,nlambda)

## GX geometric quantities
phiMax=tgridmax
phiMin=-tgridmax
pMax=(iota-nNormal)*phiMax
pMin=(iota-nNormal)*phiMin
zMax=phiMax
zMin=phiMin
denMin=phiMin-rVMEC*etabar*np.sin(pMin)
denMax=phiMax-rVMEC*etabar*np.sin(pMax)
iotaN=iota-nNormal
def tgridGX(theta,zz):
	phi=Phi(theta)
	return zz-(iotaN*(-(phiMin*zMax) + phi*(zMax - zMin) + phiMax*zMin) + etabar*rVMEC*((-zMax + zMin)*np.sin(iotaN*phi) - zMin*np.sin(iotaN*phiMax) + zMax*np.sin(iotaN*phiMin)))/(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin)))
gradparGX = Aminor*(2*iotaN*pi*(zMax - zMin))/(Laxis*(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin))))
def thetaGXgrid(zz):
	sol = fsolve(tgridGX, 0.9*zz*iotaN, args=zz)
	return sol[0]
zGXgrid=np.linspace(zMin, zMax, 2*ntgrid)
paramThetaGX = [thetaGXgrid(zz) for zz in zGXgrid]

#Output to GS2 grid (bottleneck, thing that takes longer to do, needs to be more pythy)
nz=ntgrid*2
paramTheta = np.multiply(np.linspace(-tgridmax,tgridmax,nz),(iota-nNormal))

open(gs2gridNA, 'w').close()
f = open(gs2gridNA, "w")
f.write("nlambda\n"+str(nlambda)+"\nlambda")
for item in lambdavec:
	f.write("\n%s" % item)
f.write("\nntgrid nperiod ntheta drhodpsi rmaj shat kxfac q")
f.write("\n"+str(ntgrid)+" "+str(nperiod)+" "+str(nz-1)+" "+str(drhodpsi)+" "+str(rmaj)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota))
f.write("\ngbdrift gradpar grho tgrid")
for zz in paramTheta:
	f.write("\n"+str(gbdriftNew(zz))+" "+str(gradparNew(zz))+" 1.0 "+str(zz/(iota-nNormal)))
f.write("\ncvdrift gds2 bmag tgrid")
for zz in paramTheta:
	f.write("\n"+str(cvdriftNew(zz))+" "+str(gds2New(zz))+" "+str(bmagNew(zz))+" "+str(zz/(iota-nNormal)))
f.write("\ngds21 gds22 tgrid")
for zz in paramTheta:
	f.write("\n"+str(gds21New(zz))+" "+str(gds22New(zz))+" "+str(zz/(iota-nNormal)))
f.write("\ncvdrift0 gbdrift0 tgrid")
for zz in paramTheta:
	f.write("\n"+str(cvdrift0New(zz))+" "+str(gbdrift0New(zz))+" "+str(zz/(iota-nNormal)))
f.write("\nRplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.write("\nZplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.write("\naplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.close()

#Output to GX grid (bottleneck, thing that takes longer to do, needs to be more pythy)
open(gxgridNA, 'w').close()
f = open(gxgridNA, "w")
f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
f.write("\n"+str(ntgrid)+" "+str(nperiod)+" "+str(nz-1)+" "+str(drhodpsi)+" "+str(rmaj)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota)+" 1.0")
f.write("\ngbdrift gradpar grho tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(gbdriftNew(zz))+" "+str(gradparGX)+" 1.0 "+str(zGXgrid[count]))
f.write("\ncvdrift gds2 bmag tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(cvdriftNew(zz))+" "+str(gds2New(zz))+" "+str(bmagNew(zz))+" "+str(zGXgrid[count]))
f.write("\ngds21 gds22 tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(gds21New(zz))+" "+str(gds22New(zz))+" "+str(zGXgrid[count]))
f.write("\ncvdrift0 gbdrift0 tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(cvdrift0New(zz))+" "+str(gbdrift0New(zz))+" "+str(zGXgrid[count]))
f.close()
