#!/opt/local/bin/python3

import sys
import os
from os import path
import math
import numpy as np
from scipy.io import netcdf
from qsc.qsc import Qsc
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import quad, cumtrapz
from statistics import median
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

start_time = time.time()

pi = math.pi
mu0 = 4*pi*10**(-7)
Nphi=250 # resolution for the sigma function

nperiod  = 1
drhodpsi = 1.0
rmaj     = 1.0
kxfac    = 1.0

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# read input parameters
etabar        = eval(sys.argv[1])
B0            = eval(sys.argv[2])
VMECfileIn    = sys.argv[3]
stell         = sys.argv[4]
fileIn        = sys.argv[5]
saveStellFile = sys.argv[6]
gs2gridNA     = sys.argv[7]
gxgridNA      = sys.argv[8]

# read VMEC file
f = netcdf.netcdf_file(VMECfileIn,'r',mmap=False)
raxis = f.variables['raxis_cc'][()]
zaxis = f.variables['zaxis_cs'][()]
NFP   = f.variables['nfp'][()]
Aminor = abs(f.variables['Aminor_p'][()])
phiVMEC = f.variables['phi'][()]
phiEDGE = abs(phiVMEC[-1])/(2*pi)

# obtain near-axis configuration
def chop(expr, *, max=0.3):
    return [i if i > max else 0 for i in expr]
stel = Qsc(rc=raxis,zs=-zaxis, nfp=NFP, etabar=etabar, nphi=Nphi)
iota = abs(stel.iota)
if stell=='Drevlak' or stell=='NZ1988':
	iota=-iota
sigmaSol = stel.sigma
Laxis = stel.axis_length
sprime = stel.d_l_d_phi
curvature = stel.curvature
phi = stel.phi
nNormal = stel.iotaN - stel.iota
varphi = stel.varphi

# obtain external parameters from VMEC2GS2
with open(fileIn) as f:
    content = f.readlines()
content = [x.strip() for x in content]

tgridTemp=content[0].split()
tgrid=[float(i) for i in tgridTemp[3:]]
nz=len(tgrid)
ntgrid=int(np.floor(nz/2))

shatTemp=content[11].split()
shat=float(shatTemp[1])

alphaTemp=content[12].split()
alphaVMEC=float(alphaTemp[1])

nlambdaTemp=content[13].split()
nlambda=int(nlambdaTemp[1])

normalizedtorFluxTemp=content[14].split()
normalizedtorFlux=float(normalizedtorFluxTemp[1])

paramTheta = np.multiply(tgrid,(iota-nNormal))

# find sigma and sprime functions for this specific theta grid
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

## Module to incorporate transformation from var phi to phi
# num=[tt for tt in paramTheta]
# varPhiNum = np.multiply(2*pi/Laxis,cumtrapz([sprimeFunc(tt) for tt in paramTheta], num, initial=0))
# varPhiNum = np.array(varPhiNum)-median(varPhiNum)
# varphiFunc= interp1d(num,varPhiNum, kind='cubic')

# def derFuncToMin(phi):
# 	return 2*pi/Laxis*sprimeFunc(phi)
# def PhiF(theta):
# 	varPhiTheta = (theta - alphaVMEC)/(iota - nNormal)
# 	def funcToMin(phi):
# 		return varphiFunc(phi)-varPhiTheta
# 	sol = optimize.root_scalar(funcToMin, x0=varPhiTheta, fprime=derFuncToMin, method='newton')
# 	return sol.root

# geometric quantities
rVMEC 			        = -np.sqrt((2*phiEDGE*normalizedtorFlux)/B0)
def Phi(theta):         return (theta - alphaVMEC)/(iota - nNormal)
def bmagNew(theta):     return ((Aminor**2)*B0*(1+rVMEC*etabar*np.cos(theta)))/(2*phiEDGE)
def gradparNew(theta):  return  np.sign(phiVMEC[-1])*((2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)/(sprimeFunc((alphaVMEC-theta)/(iota-nNormal))*2*pi/Laxis)
def gds2New(theta):     return (((Aminor**2)*B0)/(2*phiEDGE))*((etabar**2*np.cos(theta)**2)/curvFunc(Phi(theta))**2 + (curvFunc(Phi(theta))**2*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))**2)/etabar**2)
def gds21New(theta):    return -(np.sign(phiVMEC[-1])/(2*phiEDGE))*Aminor**2*shat*((B0*etabar**2*np.cos(theta)*np.sin(theta))/curvFunc(Phi(theta))**2+(1/etabar**2)*B0*(curvFunc(Phi(theta))**2)*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))*(-np.cos(theta)+np.sin(theta)*sigma(Phi(theta))))
def gds22New(theta):    return (Aminor**2*B0*(shat**2)*((etabar**4)*np.sin(theta)**2+(curvFunc(Phi(theta))**4)*(np.cos(theta)-np.sin(theta)*sigma(Phi(theta)))**2))/(2*phiEDGE*(etabar**2)*curvFunc(Phi(theta))**2)
def gbdriftNew(theta):  return np.sign(phiVMEC[-1])*(2*np.sqrt(2)*etabar*np.cos(theta))/np.sqrt(B0/phiEDGE)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdriftNew(theta):  return gbdriftNew(theta)
def gbdrift0New(theta): return -2*np.sqrt(2)*np.sqrt(phiEDGE/B0)*shat*etabar*np.sin(theta)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdrift0New(theta): return gbdrift0New(theta)
lambdamin=((2*phiEDGE)/((Aminor**2)*B0))/(1+abs(rVMEC*etabar))
lambdamax=((2*phiEDGE)/((Aminor**2)*B0))/(1-abs(rVMEC*etabar))
lambdavec=np.linspace(lambdamin,lambdamax,nlambda)
#GX functions only for alpha=0
#phiM=max(tgrid)
#def tgridGX(theta):   return  pi*(theta-rVMEC*etabar*np.sin(theta))/((iota-nNormal)*phiM-rVMEC*etabar*np.sin((iota-nNormal)*phiM))
#gradparGX=(2*pi*pi/Laxis)/(phiM-rVMEC*etabar*np.sin((iota-nNormal)*phiM)/(iota-nNormal))
phiMax=max(tgrid)
phiMin=min(tgrid)
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
zGXgrid=np.linspace(zMin, zMax, nz)
paramThetaGX = [thetaGXgrid(zz) for zz in zGXgrid]

#Output to GS2 grid (bottleneck, thing that takes longer to do, needs to be more pythy)
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