import desc.io
import sys
from scipy.interpolate import interp1d
import numpy as np
from desc.grid import Grid
from scipy.constants import mu_0
import toml
from desc.compute.utils import cross, dot

def get_gx_arrays(zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0):
    dzeta = zeta[1] - zeta[0]
    dzeta_pi = np.pi / nzgrid
    index_of_middle = nzgrid

    gradpar_half_grid = np.zeros(2*nzgrid)
    temp_grid = np.zeros(2*nzgrid+1)
    z_on_theta_grid = np.zeros(2*nzgrid+1)
    uniform_zgrid = np.zeros(2*nzgrid+1)

    gradpar_temp = np.copy(gradpar) 
    for i in range(2*nzgrid - 1):
        gradpar_half_grid[i] = 0.5*(np.abs(gradpar_temp[i]) + np.abs(gradpar_temp[i+1]))    
    gradpar_half_grid[2*nzgrid - 1] = gradpar_half_grid[0]

    for i in range(2*nzgrid):
        temp_grid[i+1] = temp_grid[i] + dzeta * (1 / np.abs(gradpar_half_grid[i]))
    
    for i in range(2*nzgrid+1):
        z_on_theta_grid[i] = temp_grid[i] - temp_grid[index_of_middle]
    desired_gradpar = np.pi/np.abs(z_on_theta_grid[0])
    
    for i in range(2*nzgrid+1):
        z_on_theta_grid[i] = z_on_theta_grid[i] * desired_gradpar
        gradpar_temp[i] = desired_gradpar

    for i in range(2*nzgrid+1):
        uniform_zgrid[i] = z_on_theta_grid[0] + i*dzeta_pi

    final_theta_grid = uniform_zgrid
    
    bmag_gx = interp_to_new_grid(bmag,z_on_theta_grid,uniform_zgrid)
    grho_gx = interp_to_new_grid(grho,z_on_theta_grid,uniform_zgrid)
    gds2_gx = interp_to_new_grid(gds2,z_on_theta_grid,uniform_zgrid)
    gds21_gx = interp_to_new_grid(gds21,z_on_theta_grid,uniform_zgrid)
    gds22_gx = interp_to_new_grid(gds22,z_on_theta_grid,uniform_zgrid)
    gbdrift_gx = interp_to_new_grid(gbdrift,z_on_theta_grid,uniform_zgrid)
    gbdrift0_gx = interp_to_new_grid(gbdrift0,z_on_theta_grid,uniform_zgrid)
    cvdrift_gx = interp_to_new_grid(cvdrift,z_on_theta_grid,uniform_zgrid)
    cvdrift0_gx = interp_to_new_grid(cvdrift0,z_on_theta_grid,uniform_zgrid)
    gradpar_gx = gradpar_temp
    
    return uniform_zgrid ,bmag_gx, grho_gx, gradpar_gx, gds2_gx, gds21_gx, gds22_gx, gbdrift_gx, gbdrift0_gx, cvdrift_gx, cvdrift0_gx

def interp_to_new_grid(geo_array,zgrid,uniform_grid):
    geo_array_gx = np.zeros(len(geo_array))
    f = interp1d(zgrid,geo_array,kind='cubic')
    for i in range(len(uniform_grid)-1):
        if uniform_grid[i] > zgrid[-1]:
            geo_array_gx[i] = geo_array_gx[i-1]
        else:
            geo_array_gx[i] = f(np.round(uniform_grid[i],5))
    
    geo_array_gx[-1] = geo_array[-1]
    
    return geo_array_gx


# read parameters from input file
input_file = sys.argv[1]
if len(sys.argv) > 2:
    stem = input_file.split(".")[0]
    eikfile = sys.argv[2]
    eiknc = eikfile[-8:] + ".eiknc.nc"
else:
    stem = input_file.split(".")[0]
    eikfile = stem + ".eik.out"
    eiknc = stem + ".eiknc.nc"

f = toml.load(input_file)

# note: this script assumes irho=2, so rhoc = r/a
nzgrid = int(f['Dimensions']['ntheta']/2)
npol = f['Geometry']['npol']
rho = f['Geometry']['rhotor']
path = f['Geometry']['geo_file']

eq = desc.io.load(path)  # loads desc output
if hasattr(eq, '__len__'):
    eq = eq[-1]

eq_keys = [
    "iota",
    "iota_r",
    "a",
    "rho",
    "psi"
]


flux_tube_keys = ["|B|", "|grad(psi)|^2", "grad(|B|)", "grad(alpha)", "grad(psi)",
        "B", "grad(|B|)", "kappa", "B^theta", "B^zeta", "lambda_t", "lambda_z",'p_r',
        "lambda_r", "lambda", "g^rr", "g^rt", "g^rz", "g^tz", "g^tt", "g^zz",
        "e^rho", "e^theta", "e^zeta", "|B|_r", "|B|_t", "|B|_z","sqrt(g)","B_rho","B_theta","B_zeta"]

data_eq = eq.compute(eq_keys)

psi = rho**2

fi = interp1d(data_eq['rho'],data_eq['iota'])
fs = interp1d(data_eq['rho'],data_eq['iota_r'])

iotas = fi(rho)
shears = fs(rho)

zeta_center = f['Geometry'].get('zeta_center', 0.0)
alpha = f['Geometry'].get('alpha', -iotas*zeta_center)
shift_grad_alpha = f['Geometry'].get('shift_grad_alpha', True)

if not shift_grad_alpha:
   zeta_center = 0.0

zeta = np.linspace((-np.pi*npol-alpha)/np.abs(iotas),(np.pi*npol-alpha)/np.abs(iotas),2*nzgrid+1)
iota = iotas * np.ones(len(zeta))
shear = shears * np.ones(len(zeta))
thetas = iotas/np.abs(iotas)*alpha*np.ones(len(zeta)) + iota*zeta

rhoa = rho*np.ones(len(zeta))
c = np.vstack([rhoa,thetas,zeta]).T
coords = eq.compute_theta_coords(c,tol=1e-10,maxiter=50)
grid = Grid(coords)

data = eq.compute(flux_tube_keys,grid=grid)

psib = data_eq['psi'][-1]
sign_psi = psib/np.abs(psib)
sign_iota = iotas/np.abs(iotas)

#normalizations       
Lref = data_eq['a']
Bref = 2*np.abs(psib)/Lref**2
#calculate bmag
modB = data['|B|']
bmag = modB/Bref
#calculate shear
x = Lref * rho
shat = -x/iotas * shear[0]/Lref

#calculate gradpar and grho
gradpar = Lref*data['B^zeta']/modB


#calculate grad_psi and grad_alpha
grad_psi = data['grad(psi)']
grad_psi_sq = data['|grad(psi)|^2']
lmbda = data['lambda']
lmbda_r = data['lambda_r']
lmbda_t = data['lambda_t']
lmbda_z = data['lambda_z'] 
grad_alpha_r = (lmbda_r - (zeta-zeta_center)*shear)
grad_alpha_t = (1 + lmbda_t)
grad_alpha_z = (-iota+lmbda_z)


grad_alpha = (
grad_alpha_r * data["e^rho"].T
+ grad_alpha_t * data["e^theta"].T
+ grad_alpha_z * data["e^zeta"].T
).T

grho = np.sqrt(grad_psi_sq / (Lref**2 * Bref**2 * psi))

gds2 = np.array(dot(grad_alpha,grad_alpha)) * Lref**2 * psi
gds21 = -sign_iota * np.array(dot(grad_psi,grad_alpha)) * shat/Bref
gds22 = grad_psi_sq / psi * (shat/(Lref * Bref))**2


Bra = 1/data["sqrt(g)"] * (data["B_zeta"]*(1+lmbda_t) - data["B_theta"]*(lmbda_z - iota)) *data["p_r"] * 2*Bref*Lref**2/modB**4*np.sqrt(psi)*mu_0
cvdrift = np.array(dot(cross(data['B'],data['kappa']),grad_alpha))
cvdrift *= -sign_psi * 2 * Bref * Lref**2 / modB**2 * np.sqrt(psi)
gbdrift = cvdrift + Bra

gbdrift0 = np.array(dot(cross(data['B'],data['grad(|B|)']),grad_psi))
gbdrift0 *= sign_iota * sign_psi * shat * 2 / modB**3 / np.sqrt(psi)
cvdrift0 = gbdrift0


uniform_zgrid,bmag_gx, grho_gx, gradpar_gx, gds2_gx, gds21_gx, gds22_gx, gbdrift_gx, gbdrift0_gx, cvdrift_gx, cvdrift0_gx = get_gx_arrays(zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0)

path_geo = eikfile

nperiod = 1
kxfac = 1.0
f = open(path_geo, "w")
f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
f.write("\n"+str(nzgrid)+" "+str(nperiod)+" "+str(2*nzgrid)+" "+str(1.0)+" "+ str(1/Lref)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota[0]) + " " + str(2*npol-1))

f.write("\ngbdrift gradpar grho tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(gbdrift_gx[i])+" "+str(gradpar_gx[i])+ " " + str(grho_gx[i]) + " " + str(uniform_zgrid[i]))
    
f.write("\ncvdrift gds2 bmag tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(cvdrift_gx[i])+" "+str(gds2_gx[i])+ " " + str(bmag_gx[i]) + " " + str(uniform_zgrid[i]))

f.write("\ngds21 gds22 tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(gds21_gx[i])+" "+str(gds22_gx[i])+  " " + str(uniform_zgrid[i]))

f.write("\ncvdrift0 gbdrift0 tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(cvdrift0_gx[i])+" "+str(gbdrift0_gx[i])+ " " + str(uniform_zgrid[i]))
    
f.close()



