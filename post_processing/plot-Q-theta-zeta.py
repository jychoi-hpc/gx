import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
from netCDF4 import Dataset

def map_to_periodic_range(arr, low_bound, high_bound):
    range_size = high_bound - low_bound
    mapped_arr = ((arr - low_bound) % range_size) + low_bound
    return mapped_arr

marker = ['o', 'd', 's', 'v', 'x']

if sys.argv[-1].isnumeric():
  sidx = int(sys.argv[-1])
  files = sys.argv[1:-1]
else:
  sidx = 0
  files = sys.argv[1:]

i=0
theta_all = np.array([])
zeta_all = np.array([])
Q_all = np.array([])
grho_all = np.array([])
for fname in files:
    data = Dataset(fname, mode='r')
    
    t = data.groups['Grids'].variables['time'][:]
    scale = data.groups['Geometry'].variables['theta_scale'][:]
    theta = data.groups['Grids'].variables['theta'][:]*scale
    dz = (theta[1]-theta[0])/scale
    
    iota = 1/data.groups['Geometry'].variables['q'][:]
    try:
        zeta_center = data.groups['Geometry'].variables['zeta_center'][:]
    except:  # older output files don't have zeta_center, read from eik file instead
        data_geo = Dataset(fname[:-6] + "eik.nc", mode='r')
        zeta_center = data_geo.variables['zeta_center'][:]
    alpha = -iota*zeta_center
    zeta = (theta - alpha)/iota
    
    theta_p = map_to_periodic_range(theta, -np.pi, np.pi)
    zeta_p = map_to_periodic_range(zeta, -np.pi/5, np.pi/5)

    jacobian = data.groups['Geometry'].variables['jacobian'][:]
    grho = data.groups['Geometry'].variables['grho'][:]
    fluxDenom = np.sum(jacobian*grho)
    flux_fac = jacobian/fluxDenom
    
    Qt = data.groups['Diagnostics'].variables['HeatFlux_zst'][:,sidx,:]
    Q = np.mean(Qt[int(len(t)/2):,:], axis=(0))*fluxDenom
    Q_plot = Q/fluxDenom/dz
    
    #cmap = plt.get_cmap('inferno')
    #normalize = plt.Normalize(vmin=np.min(Q_plot), vmax=1.5*np.max(Q_plot))
    #normalize = colors.LogNorm(vmin=max(1e-3,min(Q_plot)), vmax=2*max(Q_plot))
    #if np.min(Q_plot) < 0:
    #    normalize = colors.TwoSlopeNorm(vmin=np.min(Q_plot), vcenter=0., vmax=np.max(Q_plot))
    #    cmap = plt.get_cmap('bwr')
    #else:
    normalize = colors.LogNorm(vmin=max(1e-3,min(Q_plot)), vmax=2*max(Q_plot))
    cmap = plt.get_cmap('inferno')
    scatter = plt.scatter(zeta_p, theta_p, c=Q_plot, cmap=cmap, norm=normalize, marker=marker[i])
    i += 1

    theta_all = np.append(theta_all, theta_p)
    zeta_all = np.append(zeta_all, zeta_p)
    Q_all = np.append(Q_all, Q)
    grho_all = np.append(grho_all, jacobian*grho)

grid_z, grid_t = np.mgrid[-np.pi/5:np.pi/5:100j, -np.pi:np.pi:100j]
from scipy.interpolate import griddata
interp_Q = griddata((zeta_all, theta_all), Q_all, (grid_z, grid_t), method='cubic', fill_value=1e-10)
interp_grho = griddata((zeta_all, theta_all), grho_all, (grid_z, grid_t), method='cubic', fill_value=1e-10)
from scipy.integrate import trapz
int_Q = trapz(trapz(interp_Q, grid_z[:,0]), grid_t[0,:])
int_grho = trapz(trapz(interp_grho, grid_z[:,0]), grid_t[0,:])
#print("Q = ", int_Q/int_grho)

#if np.min(interp_Q/int_grho) < 0:
#    normalize = colors.TwoSlopeNorm(vmin=np.min(interp_Q/int_grho), vcenter=0., vmax=np.max(interp_Q/int_grho))
#    cmap = plt.get_cmap('bwr')
#else:
normalize = colors.LogNorm(vmin=max(1e-3,np.min(Q_plot)), vmax=2*np.max(Q_plot))
cmap = plt.get_cmap('inferno')
scatter.set_norm(normalize)
scatter.set_cmap(cmap)
plt.colorbar(scatter)
pc = plt.pcolormesh(grid_z, grid_t, interp_Q/int_grho, norm=normalize, cmap=cmap, alpha=0.5)
plt.xlabel(r"$\zeta$")
plt.ylabel(r"$\theta$")
plt.title(fr"$< Q > = {int_Q/int_grho:.3f}$")
plt.show()
