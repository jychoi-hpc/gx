#!/usr/bin/env python
"""
This a Pythonized geometry module to read tokamak and stellarator equilibria from a VMEC file and caculate the geometric coefficients needed for a GX/GS2 run. Additionally, this module can vary the pressure and iota gradients self-consistently (while respecting MHD force balance) according to the work by Greene and Chance + Hegna and Nakajima and recalculate the geometry coefficients.

Dependencies:
netcdf4, pip install netcdf4
booz_xform, pip install booz_xform

For axisymmetric equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 1 <desired output name>

For 3D equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 0 <desired output name>

A portion of this script is based on Matt Landreman's vmec_geometry module for the SIMSOPT framework.
For axisymmetric equilibria, make sure that ntor > 1 in the VMEC wout file.
"""

import sys
import os

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, PPoly, CubicSpline
from scipy.integrate import cumulative_trapezoid as ctrap
from scipy.integrate import simpson as simps

print("Running pyvmec geometry module...")

try:
    from netCDF4 import Dataset as ds
except ImportError:
    print("The netCDF4 package is required. This can be installed using pip via 'pip install netCDF4'.")

try:
    import booz_xform as bxform
except ImportError:
    print("The booz_xform package is required. This can be installed using pip via 'pip install booz_xform'.")
    raise

# read parameters from input file
input_file = sys.argv[1]

# Add toml directory to module search path
parent_dir = os.path.abspath(os.path.dirname(__file__))
toml_dir = os.path.join(parent_dir, "toml")

sys.path.append(toml_dir)
import toml

if len(sys.argv) > 2:
    stem = input_file[:-3]
    eiknc = sys.argv[2]
else:
    stem = input_file[:-3]
    eiknc = stem + ".eik.nc"

f = toml.load(input_file)

ntgrid = int(f["Dimensions"]["ntheta"] / 2 + 1)
ntheta_in = int(f["Dimensions"]["ntheta"])
npol = f.get("Geometry").get("npol", 1.0)
npol_min = f.get("Geometry").get("npol_min", None)

try:
    isaxisym = f["Geometry"]["isaxisym"]
except KeyError:
    isaxisym = False

vmec_fname = f["Geometry"]["vmec_file"]

# rhoc == s = psi/psi_LCFS
try:
    rhoc = f["Geometry"]["torflux"]
except KeyError:
    rhoc = f["Geometry"]["desired_normalized_toroidal_flux"]

try:
    betaprim = f["Geometry"]["betaprim"]
except KeyError:
    # compute betaprim from tprim, fprim, and beta
    tprims = np.array(f["species"]["tprim"])
    fprims = np.array(f["species"]["fprim"])
    ns = np.array(f["species"]["dens"])
    Ts = np.array(f["species"]["temp"])
    beta_gx = f["Physics"]["beta"]

    betaprim = -beta_gx * np.sum(ns * Ts * (tprims + fprims))

try:
    boundary = f["Domain"]["boundary"]
except KeyError:
    boundary = "linked"

if boundary == "exact periodic":
    flux_tube_cut = "gds21"
elif boundary == "continuous drifts":
    flux_tube_cut = "gbdrift0"
elif boundary == "fix aspect":
    flux_tube_cut = "aspect"
else:
    flux_tube_cut = "none"

y0 = f.get("Domain").get("y0", 10.0)
x0 = f.get("Domain").get("x0", y0)

jtwist_in = f.get("Domain").get("jtwist", None)
jtwist_max = f.get("Domain").get("jtwist_max", None)

default = -1
if npol_min is not None:
    default = 0
which_crossing = f.get("Geometry").get("which_crossing", default)

# include self-consistent equilibrium variation due to shear and pressure gradient 
include_shear_variation = f.get("Geometry").get("include_shear_variation", False)
include_pressure_variation = f.get("Geometry").get("include_pressure_variation", False)

mu_0 = 4 * np.pi * (1.0e-7)

################################################################################
########--------------------HELPER FUNCTIONS------------------------############
################################################################################


def nperiod_set(arr, npol, extend=True, brr=None):
    """
    Contract or extend a large array to a smaller one.

    Truncates (or extends) an array to a smaller one. This function gives us the ability to truncate a variable to theta in [-pi, pi].

    Inputs
    ------
    arr: numpy array
    Input array of the dependent variable
    brr: numpy array
    Input array of the independent variable
    extend: boolean
    Whether to extend instead of contract a large array

    Returns
    -------
    A truncated or extended array arr
    """
    if extend is True and npol > 1:
        arr_temp0 = arr - (arr[0] + npol * np.pi)
        arr_temp1 = arr_temp0
        for i in np.arange(1, npol):
            arr_temp1 = np.concatenate((arr_temp1, arr_temp0[1:] + 2 * np.pi * i))

        arr = arr_temp1
    elif brr is None:  # contract the theta array
        eps = 1e-11
        arr_temp0 = arr[arr <= npol * np.pi + eps]
        arr_temp0 = arr_temp0[arr_temp0 >= -npol * np.pi - eps]
        arr = arr_temp0
    else:  # contract the non-theta array using the theta array brr
        eps = 1e-11
        arr_temp0 = arr[brr <= npol * np.pi + eps]
        brr_temp0 = brr[brr <= npol * np.pi + eps]
        arr_temp0 = arr_temp0[brr_temp0 >= -npol * np.pi - eps]
        brr_temp0 = brr_temp0[brr_temp0 >= -npol * np.pi - eps]
        arr = arr_temp0

    return arr


def dermv(arr, brr, par="e"):
    """
    Finite difference subroutine for a non-uniform grid.
    Calculates d(arr)/d(brr) using a second-order finite difference method.
    brr can be a non-uniformly spaced array
    Inputs
    ------
    arr: numpy array
    Input array of the dependent variable
    brr: numpy array
    Input array of the independent variable
    par: str
    Expected parity of the output array, par = 'e' means even par = 'o' means odd

    Returns
    -------
    The derivative d(arr)/d(brr)
    """
    temp = np.shape(arr)
    if len(temp) == 1:  # One dimensional input array
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            # diff_arr[0, 1:-1] = (
            #     np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            # )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))

            h1 = np.abs(brr[0, 1]) - np.abs(brr[0, 0])
            h0 = np.abs(brr[0, -1]) - np.abs(brr[0, -2])
            diff_arr[0, 0] = (4 * arr[0, 1] - 3 * arr[0, 0] - arr[0, 2]) / (
                2 * (brr[0, 1] - brr[0, 0])
            )

            diff_arr[0, -1] = (-4 * arr[0, -2] + 3 * arr[0, -1] + arr[0, -3]) / (
                2 * (brr[0, -1] - brr[0, -2])
            )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        diff_arr = diff_arr[0]

    else:
        d1, d2, d3 = np.shape(arr)
        diff_arr = np.zeros((d1, d2, d3))
        if par == "e":  # Even parity
            diff_arr[:, :, 0] = np.zeros((d1, d2))
            diff_arr[:, :, -1] = np.zeros((d1, d2))
            for i in range(1, d3 - 1):
                h1 = brr[:, :, i + 1] - brr[:, :, i]
                h0 = brr[:, :, i] - brr[:, :, i - 1]
                diff_arr[:, :, i] = (
                    arr[:, :, i + 1] / h1**2
                    + arr[:, :, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[:, :, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            diff_arr[:, :, 0] = (
                2 * (arr[:, :, 1] - arr[:, :, 0]) / (2 * (brr[:, :, 1] - brr[:, :, 0]))
            )
            diff_arr[:, :, -1] = (
                2
                * (arr[:, :, -1] - arr[:, :, -2])
                / (2 * (brr[:, :, -1] - brr[:, :, -2]))
            )
            for i in range(1, d3 - 1):
                h1 = brr[:, :, i + 1] - brr[:, :, i]
                h0 = brr[:, :, i] - brr[:, :, i - 1]
                diff_arr[:, :, i] = (
                    arr[:, :, i + 1] / h1**2
                    + arr[:, :, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[:, :, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)

    return diff_arr


#################################################################################
##############---------------EQUILIBRIUM CALC.------------------#################
#################################################################################


class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """


def vmec_splines(nc_obj, booz_obj):
    """
    Initialize radial splines for a VMEC equilibrium.

    Args:
        vmec: a netCDF object

    Returns:
        A structure with the splines as attributes.
    """
    results = Struct()

    rmnc_b = []
    zmns_b = []
    numns_b = []

    d_rmnc_b_d_s = []
    d_zmns_b_d_s = []
    d_numns_b_d_s = []

    ns = nc_obj.variables["ns"][:].data
    s_full_grid = np.linspace(0, 1, ns)
    #s_half_grid = s_full_grid[1:] - 0.5 * np.diff(s_full_grid)[0]
    s_half_grid = 0.5*(s_full_grid[0:-1] + s_full_grid[1:])

    # Boozer quantities are calculated on the half grid by booz_xform
    for jmn in range(int(booz_obj.mnboz)):
        rmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.rmnc_b.T[:, jmn])
        )
        zmns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.zmns_b.T[:, jmn])
        )
        numns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.numns_b.T[:, jmn])
        )

        d_rmnc_b_d_s.append(rmnc_b[-1].derivative())
        d_zmns_b_d_s.append(zmns_b[-1].derivative())
        d_numns_b_d_s.append(numns_b[-1].derivative())

    gmnc_b = []
    bmnc_b = []
    d_bmnc_b_d_s = []

    for jmn in range(int(booz_obj.mnboz)):
        gmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.gmnc_b.T[:, jmn])
        )
        bmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.bmnc_b.T[:, jmn])
        )
        d_bmnc_b_d_s.append(bmnc_b[-1].derivative())

    results.Gfun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_G)
    results.Ifun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_I)

    # Useful 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["pres"][1:]
    )
    results.d_pressure_d_s = results.pressure.derivative()
    results.psi = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["phi"][1:] / (2 * np.pi)
    )
    results.d_psi_d_s = results.psi.derivative()
    results.iota = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["iotas"][1:]
    )
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = nc_obj.variables["phi"][-1].data
    variables = ["Aminor_p", "nfp", "raxis_cc", "mpol", "ntor"]
    for v in variables:
        results.__setattr__(v, eval("nc_obj.variables['" + v + "'][:].data"))

    variables1 = ["xm_b", "xn_b", "xm_nyq_b", "xn_nyq_b", "mnbooz", "mboz", "nboz"]
    variables2 = ["xm_b", "xn_b", "xm_b", "xn_b", "mnboz", "mboz", "nboz"]
    for k, v in enumerate(variables1):
        results.__setattr__(v, eval("booz_obj." + variables2[k]))

    variables = [
        "rmnc_b",
        "zmns_b",
        "numns_b",
        "d_rmnc_b_d_s",
        "d_zmns_b_d_s",
        "d_numns_b_d_s",
        "gmnc_b",
        "bmnc_b",
        "d_bmnc_b_d_s",
    ]
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#########################################################################################################
#######################------------------GEOMETRY CALCULATION FUN--------------------####################
#########################################################################################################


def vmec_fieldlines(
    vmec_fname,
    s,
    betaprim,
    toml_dict,
    theta1d=None,
    phi1d=None,
    isaxisym=False,
    res_theta=201,
    res_phi=201,
):
    """
    Geometry routine for GX/GS2.

    Takes in a 1D theta or phi array in boozer coordinates, an array of flux surfaces,
    another array of field line labels and generates the coefficients needed for a local
    stability analysis.
    Additionally, this routinerecalculates the geometric coefficients if the user to wants to vary self-consistently the local pressure gradient and average shear and.
    Inputs:
    ------
    s: List or numpy array
    The normalized toroidal flux psi/psi_boundary
    alpha: list or numpy array
    alpha = theta_b - iota * phi_b is the field line label
    theta1d: numpy array
    Boozer theta
    phi1d: numpy array
    Boozer phi

    Outputs
    -------
    gds22: numpy array
    Flux expansion term
    gds21 numpy array
    Integrated local shear
    gds2: numpy array
    Field line bending
    bmag: numpy array
    normalized magnetic field strength
    gradpar: numpy array
    Parallel gradient b dot grad phi
    gbdrift: numpy array
    Grad-B drift geometry factor
    cvdrift: numpy array
    Curvature drift geometry factor
    cvdrift0: numpy array
    theta_PEST: numpy array
    theta_PEST for the given theta_b array.
    theta_geo: numpy array
    geometric (arctan) theta for the given theta_b array.
    """
    nc_obj = ds(vmec_fname, "r")

    mpol = nc_obj.variables["mpol"][:].data
    ntor = nc_obj.variables["ntor"][:].data

    booz_obj = bxform.Booz_xform()
    booz_obj.verbose = 0
    booz_obj.read_wout(vmec_fname)
    booz_obj.mboz = int(2 * mpol)
    booz_obj.nboz = int(2 * ntor)
    booz_obj.run()

    vs = vmec_splines(nc_obj, booz_obj)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)


    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    d_psi_d_s = vs.d_psi_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2 * s / iota) * d_iota_d_s  # depends on the definitn of rho
    sqrt_s = np.sqrt(s)

    print("iota = ", iota)
    print("shat = ", shat)
    print("d iota / ds", d_iota_d_s)
    print("d pressure / ds", d_pressure_d_s)

    zeta_center = toml_dict.get("Geometry").get("zeta_center", 0.0)
    alpha = toml_dict.get("Geometry").get("alpha", -iota*zeta_center)
    shift_grad_alpha = toml_dict.get("Geometry").get("shift_grad_alpha", True)

    if not shift_grad_alpha:
        zeta_center = 0.0

    # Make sure alpha is an array
    # For axisymmetric equilibria, all field lines are identical, i.e., your choice of alpha doesn't matter
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    try:
        iota_input = toml_dict["Geometry"]["iota_input"]
    except KeyError:
        iota_input = iota

    try:
        s_hat_input = toml_dict["Geometry"]["s_hat_input"]
        if s_hat_input == 0.0:
            s_hat_input = 1.0e-8
    except KeyError:
        s_hat_input = shat

    nfp = vs.nfp

    L_reference = vs.Aminor_p

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    mnmax_b = vs.mnbooz

    G = vs.Gfun(s)
    d_G_d_s = vs.Gfun.derivative()(s)
    I = vs.Ifun(s)
    d_I_d_s = vs.Ifun.derivative()(s)

    rmnc_b = np.zeros((ns, mnmax_b))
    zmns_b = np.zeros((ns, mnmax_b))
    numns_b = np.zeros((ns, mnmax_b))
    d_rmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_zmns_b_d_s = np.zeros((ns, mnmax_b))
    d_numns_b_d_s = np.zeros((ns, mnmax_b))

    numns_b = np.zeros((ns, mnmax_b))
    gmnc_b = np.zeros((ns, mnmax_b))
    bmnc_b = np.zeros((ns, mnmax_b))
    d_bmnc_b_d_s = np.zeros((ns, mnmax_b))

    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    theta_b = np.zeros((ns, nalpha, nl))
    phi_b = np.zeros((ns, nalpha, nl))

    Vprime = np.zeros((ns, 1))

    for jmn in range(mnmax_b):
        rmnc_b[:, jmn] = vs.rmnc_b[jmn](s)
        zmns_b[:, jmn] = vs.zmns_b[jmn](s)
        numns_b[:, jmn] = vs.numns_b[jmn](s)
        d_rmnc_b_d_s[:, jmn] = vs.d_rmnc_b_d_s[jmn](s)
        d_zmns_b_d_s[:, jmn] = vs.d_zmns_b_d_s[jmn](s)
        d_numns_b_d_s[:, jmn] = vs.d_numns_b_d_s[jmn](s)
        gmnc_b[:, jmn] = vs.gmnc_b[jmn](s)
        bmnc_b[:, jmn] = vs.bmnc_b[jmn](s)
        d_bmnc_b_d_s[:, jmn] = vs.d_bmnc_b_d_s[jmn](s)

    if theta1d is None:
        # We are given phi_boozer. Compute theta_boozer
        for js in range(ns):
            phi_b[js, :, :] = phi1d[None, :]
            theta_b[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :])
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_b[js, :, :] = theta1d[None, :]
            phi_b[js, :, :] = (theta1d[None, :] - alpha[:, None]) / iota[js]

    # Now that we know theta_boozer, compute all the geometric quantities
    angle_b = (
        xm_b[:, None, None, None] * (theta_b[None, :, :, :])
        - xn_b[:, None, None, None] * phi_b[None, :, :, :]
    )
    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)

    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)

    flipit = 0.0

    if isaxisym == 1:
        # if R is increasing AND Z is decreasing, we must be moving counter clockwise from
        # the inboard side, otherwise we need to flip the theta coordinate
        if R_b[0][0][0] > R_b[0][0][1] or Z_b[0][0][1] > Z_b[0][0][0]:
            flipit = 1
    else:  # we disable flipit
        flipit = 0

    R_mag_ax = vs.raxis_cc[0]

    #####################################################################################
    #####################------------BOOZER CALCULATIONS--------------###################
    #####################################################################################

    if flipit == 1:
        angle_b = (
            xm_b[:, None, None, None] * (theta_b[None, :, :, :] + np.pi)
            - xn_b[:, None, None, None] * phi_b
        )
    else:
        angle_b = (
            xm_b[:, None, None, None] * theta_b[None, :, :, :]
            - xn_b[:, None, None, None] * phi_b
        )

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b
    # Order of indices in cosangle_b and sinangle_b: mn_b, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn_b
    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    d_R_b_d_s = np.einsum("ij,jikl->ikl", d_rmnc_b_d_s, cosangle_b)
    d_R_b_d_theta_b = -np.einsum("ij,jikl->ikl", rmnc_b, msinangle_b)
    d_R_b_d_phi_b = np.einsum("ij,jikl->ikl", rmnc_b, nsinangle_b)

    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)
    d_Z_b_d_s = np.einsum("ij,jikl->ikl", d_zmns_b_d_s, sinangle_b)
    d_Z_b_d_theta_b = np.einsum("ij,jikl->ikl", zmns_b, mcosangle_b)
    d_Z_b_d_phi_b = -np.einsum("ij,jikl->ikl", zmns_b, ncosangle_b)

    nu_b = np.einsum("ij,jikl->ikl", numns_b, sinangle_b)
    d_nu_b_d_s = np.einsum("ij,jikl->ikl", d_numns_b_d_s, sinangle_b)
    d_nu_b_d_theta_b = np.einsum("ij,jikl->ikl", numns_b, mcosangle_b)
    d_nu_b_d_phi_b = -np.einsum("ij,jikl->ikl", numns_b, ncosangle_b)

    # sqrt_g_booz = (G + iota * I)/B**2
    sqrt_g_booz = np.einsum("ij,jikl->ikl", gmnc_b, cosangle_b)
    d_sqrt_g_booz_d_theta_b = -np.einsum("ij,jikl->ikl", gmnc_b, msinangle_b)
    d_sqrt_g_booz_d_phi_b = np.einsum("ij,jikl->ikl", gmnc_b, nsinangle_b)
    modB_b = np.einsum("ij,jikl->ikl", bmnc_b, cosangle_b)
    d_B_b_d_s = np.einsum("ij,jikl->ikl", d_bmnc_b_d_s, cosangle_b)

    Vprime = gmnc_b[:, 0]

    delmnc_b[:, 1:] = gmnc_b[:, 1:] / Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        * 1
        / edge_toroidal_flux_over_2pi
        * mu_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / (xm_b[1:] * iota[:, None] - xn_b[1:])
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * I[:, None])
        / (
            (xm_b[1:] * iota[:, None] - xn_b[1:])
            * (G[:, None] + iota[:, None] * I[:, None])
        )
    )

    beta_b = np.einsum("ij,jikl->ikl", betamns_b, sinangle_b)
    lambda_b = np.einsum("ij,jikl->ikl", lambmnc_b, cosangle_b)

    ###################################################################
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # This calculation is done in Boozer coordinates
    ####################################################################
    phi_cyl = phi_b - nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)
    # X = R * cos(phi):
    d_X_d_theta_b = d_R_b_d_theta_b * cosphi - R_b * sinphi * (-1 * d_nu_b_d_theta_b)
    d_X_d_phi_b = d_R_b_d_phi_b * cosphi - R_b * sinphi * (1 - d_nu_b_d_phi_b)
    d_X_d_s = d_R_b_d_s * cosphi - R_b * sinphi * (-1 * d_nu_b_d_s)
    # Y = R * sin(phi):
    d_Y_d_theta_b = d_R_b_d_theta_b * sinphi + R_b * cosphi * (-1 * d_nu_b_d_theta_b)
    d_Y_d_phi_b = d_R_b_d_phi_b * sinphi + R_b * cosphi * (1 - d_nu_b_d_phi_b)
    d_Y_d_s = d_R_b_d_s * sinphi + R_b * cosphi * (-1 * d_nu_b_d_s)

    # Dual relations
    grad_psi_X = (
        d_Y_d_theta_b * d_Z_b_d_phi_b - d_Z_b_d_theta_b * d_Y_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Y = (
        d_Z_b_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_b_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Z = (
        d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b
    ) / sqrt_g_booz

    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2

    # Check varible names
    grad_theta_b_X = (d_Y_d_phi_b * d_Z_b_d_s - d_Z_b_d_phi_b * d_Y_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Y = (d_Z_b_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_b_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Z = (d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )

    check1 = (
        grad_theta_b_X * d_X_d_theta_b
        + grad_theta_b_Y * d_Y_d_theta_b
        + grad_theta_b_Z * d_Z_b_d_theta_b
    )
    check2 = (
        grad_psi_X * d_X_d_s + grad_psi_Y * d_Y_d_s + grad_psi_Z * d_Z_b_d_s
    ) / edge_toroidal_flux_over_2pi

    grad_phi_b_X = (d_Y_d_s * d_Z_b_d_theta_b - d_Z_b_d_s * d_Y_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Y = (d_Z_b_d_s * d_X_d_theta_b - d_X_d_s * d_Z_b_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Z = (d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )

    grad_alpha_X = (
        -(phi_b-zeta_center) * d_iota_d_s[:, None, None] * grad_psi_X / edge_toroidal_flux_over_2pi
        + grad_theta_b_X
        - iota[:, None, None] * grad_phi_b_X
    )
    grad_alpha_Y = (
        -(phi_b-zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Y / edge_toroidal_flux_over_2pi
        + grad_theta_b_Y
        - iota[:, None, None] * grad_phi_b_Y
    )
    grad_alpha_Z = (
        -(phi_b-zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Z / edge_toroidal_flux_over_2pi
        + grad_theta_b_Z
        - iota[:, None, None] * grad_phi_b_Z
    )

    #####################################################################################
    ##############------------LOCAL VARIATION OF A 3D EQUILIBRIUM------------############
    #####################################################################################
    # Calculating the coefficients D1 and D2 needed for Hegna-Nakajima calculation
    # NOTE: 2D functions do not require the alpha dimension. Remove it later.
    # NOTE: This calculation needs to be wrapped in a loop over ns (flux surfaces)
    ## Full flux surface average of various quantities needed to calculate D_HNGC
    ntheta_grid = res_theta
    nphi_grid = res_phi
    theta_b_grid = np.linspace(-np.pi, np.pi, ntheta_grid)
    phi_b_grid = np.linspace(-np.pi, np.pi, nphi_grid)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)

    if flipit == 1:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :] + np.pi)
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )
    else:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :])
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)

    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = np.einsum("ij,jiklm->iklm", lambmnc_b, cosangle_b_2D)

    R_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, cosangle_b_2D)
    d_R_b_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, msinangle_b_2D)
    d_R_b_d_phi_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, nsinangle_b_2D)

    d_Z_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, mcosangle_b_2D)
    d_Z_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", zmns_b, ncosangle_b_2D)

    nu_b_2D = np.einsum("ij,jiklm->iklm", numns_b, sinangle_b_2D)
    d_nu_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", numns_b, mcosangle_b_2D)
    d_nu_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", numns_b, ncosangle_b_2D)

    sqrt_g_booz_2D = np.einsum("ij,jiklm->iklm", gmnc_b, cosangle_b_2D)
    modB_b_2D = np.einsum("ij,jiklm->iklm", bmnc_b, cosangle_b_2D)

    #########################################################################
    # We repeat the above exercise to calculate R and Z but use a 2D
    # (theta, phi) grid. This is used to calculate the deformation
    # coefficients that give us the local equilibrium variation
    #########################################################################
    ph_nat_2D = ph_b_2D - nu_b_2D
    sinphi_2D = np.sin(ph_nat_2D)
    cosphi_2D = np.cos(ph_nat_2D)
    # X = R * cos(phi):
    d_X_d_th_b_2D = d_R_b_d_theta_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_X_d_phi_2D = d_R_b_d_phi_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )
    # Y = R * sin(phi):
    d_Y_d_th_b_2D = d_R_b_d_theta_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_Y_d_phi_2D = d_R_b_d_phi_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )

    grad_psi_X_2D = (
        d_Y_d_th_b_2D * d_Z_b_d_phi_b_2D - d_Z_b_d_theta_b_2D * d_Y_d_phi_2D
    ) / sqrt_g_booz_2D
    grad_psi_Y_2D = (
        d_Z_b_d_theta_b_2D * d_X_d_phi_2D - d_X_d_th_b_2D * d_Z_b_d_phi_b_2D
    ) / sqrt_g_booz_2D
    grad_psi_Z_2D = (
        d_X_d_th_b_2D * d_Y_d_phi_2D - d_Y_d_th_b_2D * d_X_d_phi_2D
    ) / sqrt_g_booz_2D

    g_sup_psi_psi_2D = grad_psi_X_2D**2 + grad_psi_Y_2D**2 + grad_psi_Z_2D**2
    g_sup_psi_psi_2D_inv = 1 / g_sup_psi_psi_2D

    lam_over_g_sup_psi_psi_2D = lambda_b_2D * g_sup_psi_psi_2D_inv

    # Flux surface integrals D1 and D2 are needed to locally vary the gradients of a 3D equilibrium.
    D1 = (
        simps(
            [
                simps(g_sup_psi_psi_1D_inv, x=theta_b_grid)
                for g_sup_psi_psi_1D_inv in g_sup_psi_psi_2D_inv[0][0]
            ],
            x=phi_b_grid,
        )
        / (2 * np.pi) ** 2
    )

    D2 = (
        simps(
            [
                simps(lam_over_g_sup_psi_psi_1D, x=theta_b_grid)
                for lam_over_g_sup_psi_psi_1D in lam_over_g_sup_psi_psi_2D[0][0]
            ],
            x=phi_b_grid,
        )
        / (2 * np.pi) ** 2
    )

    ## EQUILIBRIUM CHECK: Flux surface averaged MHD force balance.
    residual = d_G_d_s[:, None, None] \
        + iota[:, None, None] * d_I_d_s[:, None, None] \
        + mu_0 * d_pressure_d_s[:, None, None] * Vprime

    check = np.allclose(
        residual,
        1e-7,
        atol=5e-3,
    )
    if not check:
        print(f"WARNING: MHD force balance not exactly satisfied. Error = {residual}")

    # integrated inverse flux expansion term
    intinv_g_sup_psi_psi = ctrap(1 / g_sup_psi_psi, phi_b, initial=0)
    int_lambda_div_g_sup_psi_psi = ctrap(lambda_b / g_sup_psi_psi, phi_b, initial=0)

    # This theta_0 should always be 0
    theta_0 = 0
    spl0 = InterpolatedUnivariateSpline(theta_b[0][0], intinv_g_sup_psi_psi[0][0])
    intinv_g_sup_psi_psi = intinv_g_sup_psi_psi - spl0(theta_0)

    spl1 = InterpolatedUnivariateSpline(
        theta_b[0][0], int_lambda_div_g_sup_psi_psi[0][0]
    )
    int_lambda_div_g_sup_psi_psi = int_lambda_div_g_sup_psi_psi - spl1(theta_0)

    # Additional shear (in addn. to the nominal vals)
    d_iota_d_s_1 = (
        -(iota_input / (2 * s)) * s_hat_input + (iota / (2 * s)) * shat
    ) * np.ones((ns,))
    sfac = shat / s_hat_input

    if include_shear_variation == False:
        d_iota_d_s_1 = 0*d_iota_d_s_1
        sfac = 1

    # NOTE: Compare beta definitions
    # This is half of the total beta_N. Used in GS2 as beta_ref
    beta_N = 4 * np.pi * 1e-7 * vs.pressure(s) / B_reference**2

    # Additional pressure (in addn. to the nominal vals)
    d_pressure_d_s_1 = ( (betaprim / (4 * np.sqrt(s)) * B_reference**2 * np.ones((ns,)))
        - mu_0 * d_pressure_d_s * np.ones((ns,))
    )

    if d_pressure_d_s == 0:
        d_pressure_d_s = 1e-8 * np.ones((ns,))[:, None, None]

    pfac = (
        betaprim * B_reference**2 / (4 * np.sqrt(s))
        / (mu_0 * d_pressure_d_s)
    )

    if include_pressure_variation == False:
        pfac = 1
        d_pressure_d_s_1 = 0*d_pressure_d_s_1

    print(f"pfac = {pfac}")
    print(f"sfac = {sfac}")

    # The deformation term from Hegna-Nakajima and Green-Chance papers
    D_HNGC = (
        1
        / edge_toroidal_flux_over_2pi
        * (
            d_iota_d_s_1[:, None, None] * (intinv_g_sup_psi_psi / D1 - phi_b + zeta_center)
            - d_pressure_d_s_1[:, None, None]
            * Vprime[:, None, None]
            * (G[:, None, None] + iota[:, None, None] * I[:, None, None])
            * (int_lambda_div_g_sup_psi_psi - D2 * intinv_g_sup_psi_psi / D1)
        )
    )

    # Now we recalculate some of the geometric coefficients in Boozer coordinates
    # Partially calculated in boozer coordinates
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    # Intergrated local shear L1 is calculated using covariant basis
    # expressions in Hegna and Nakajima
    # We remove the secular part form the integrated local shear grad_alpha_dot_grad_psi_alt
    # NOTE: Potential sign issue here
    L0 = -1 * (
        grad_alpha_dot_grad_psi / g_sup_psi_psi
        + 1 / edge_toroidal_flux_over_2pi * d_iota_d_s[:, None, None] * (phi_b-zeta_center)
    )
    # L1 is the integrated local shear
    L1 = (
        -1 / edge_toroidal_flux_over_2pi * d_iota_d_s_1[:, None, None] * (phi_b-zeta_center)
        + grad_alpha_dot_grad_psi / g_sup_psi_psi
        - D_HNGC
    )

    L2 = d_iota_d_s[:, None, None] * 1 / edge_toroidal_flux_over_2pi

    # Normal curvature
    # NOTE: Test a case close to a rational surface to check the sign of beta_b
    kappa_n = (
        1
        / modB_b**2
        * (modB_b * d_B_b_d_s + mu_0 * d_pressure_d_s[:, None, None])
        * 1
        / edge_toroidal_flux_over_2pi
        - beta_b
        / (
            2
            * sqrt_g_booz
            * (G[:, None, None] + iota[:, None, None] * I[:, None, None])
        )
        * d_sqrt_g_booz_d_phi_b
        + L0
        * (G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b)
        / (2 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))
    )

    # Geodesic curvature
    kappa_g = (
        G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b
    ) / (2 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))

    B_cross_kappa_dot_grad_alpha_b = (kappa_n + kappa_g * L1) * modB_b**2

    B_cross_kappa_dot_grad_psi_b = kappa_g * modB_b**2

    grad_alpha_dot_grad_alpha_b = modB_b**2 / g_sup_psi_psi + g_sup_psi_psi * L1**2
    grad_alpha_dot_grad_psi_b = g_sup_psi_psi * L1
    grad_psi_dot_grad_psi_b = (
        g_sup_psi_psi * L2
    )  # This is wrong. L2 should be different

    ## Now we calculate the same set of quantities in boozer coordinates after varying the
    ## local gradients.
    bmag = modB_b / B_reference
    gradpar_theta_b = -L_reference / modB_b * 1 / sqrt_g_booz * iota[:, None, None]
    gradpar_theta_PEST = (
        -L_reference
        * iota[:, None, None]
        * 1
        / modB_b
        * 1
        / sqrt_g_booz
        * (1 - d_nu_b_d_theta_b)
    )
    gradpar_phi = L_reference / modB_b * 1 / sqrt_g_booz

    gds2 = grad_alpha_dot_grad_alpha_b * L_reference * L_reference * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi_b * sfac * shat[:, None, None] / B_reference
    gds22 = (
        g_sup_psi_psi
        * (sfac * shat[:, None, None]) ** 2
        / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])
    )

    grho = np.sqrt(
        g_sup_psi_psi
        / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])
    )

    gbdrift0 = (
        -1.0
        * B_cross_kappa_dot_grad_psi_b
        * 2
        * sfac
        * shat[:, None, None]
        / (modB_b * modB_b * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift0 = gbdrift0

    cvdrift = (
        -1.0
        * 2
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * B_cross_kappa_dot_grad_alpha_b
        / (modB_b * modB_b)
        * toroidal_flux_sign
    )

    gbdrift = cvdrift + 2 * B_reference * L_reference * L_reference * sqrt_s[
        :, None, None
    ] * mu_0 * pfac * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (
        edge_toroidal_flux_over_2pi * modB_b * modB_b
    )

    cvdrift0 = gbdrift0
    # PEST theta; useful for comparison
    theta_PEST = theta_b - iota * nu_b

    # geometric theta; denotes the actual poloidal angle
    theta_geo = np.arctan2(Z_b, R_b - R_mag_ax)

    int_loc_shr = L0 + L1 + L2
    # Package results into a structure to return:
    results = Struct()
    variables = [
        "iota_input",
        "d_iota_d_s",
        "d_pressure_d_s",
        "d_psi_d_s",
        "s_hat_input",
        "alpha",
        "theta_b",
        "phi_b",
        "theta_PEST",
        "theta_geo",
        "edge_toroidal_flux_over_2pi",
        "R_b",
        "Z_b",
        "betaprim",
        "bmag",
        "gradpar_theta_b",
        "gradpar_theta_PEST",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
        "grho",
        "alpha",
        "zeta_center",
        "nfp"
    ]

    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#############################################################################
########-----------------CALCULATING GEOMETRY----------------################
#############################################################################

if npol_min is not None:
    nt = ntgrid
    ntheta = ntheta_in + 1
    # double grid to allow cut after npol_min
    nt_ext = ntgrid*2
    ntheta_ext = ntheta_in*2 + 1
    # This is Boozer theta
    theta = np.linspace(-2*npol_min * np.pi, 2*npol_min * np.pi, ntheta_ext)
else:
    nt = ntgrid
    ntheta = ntheta_in + 1
    # This is Boozer theta
    theta = np.linspace(-npol * np.pi, npol * np.pi, ntheta)

kxfac = abs(1.0)

geo_coeffs = vmec_fieldlines(
    vmec_fname, rhoc, betaprim, f, theta1d=theta, isaxisym=isaxisym
)


shat = geo_coeffs.s_hat_input
qfac = abs(1 / geo_coeffs.iota_input)
bmag = geo_coeffs.bmag[0][0]
gradpar = abs(geo_coeffs.gradpar_theta_b[0][0])
cvdrift = geo_coeffs.cvdrift[0][0]
gbdrift = geo_coeffs.gbdrift[0][0]
gbdrift0 = geo_coeffs.gbdrift0[0][0]
cvdrift0 = geo_coeffs.cvdrift0[0][0]
gds2 = geo_coeffs.gds2[0][0]
gds21 = geo_coeffs.gds21[0][0]
gds22 = geo_coeffs.gds22[0][0]
R = geo_coeffs.R_b[0][0]
Z = geo_coeffs.Z_b[0][0]
grho = geo_coeffs.grho[0][0]
alpha = geo_coeffs.alpha
zeta_center = geo_coeffs.zeta_center
nfp = geo_coeffs.nfp

# rho = sqrt(psi/psi_LCFS) = sqrt(rhoc)
dpsidrho = 2 * np.sqrt(rhoc) * geo_coeffs.edge_toroidal_flux_over_2pi
drhodpsi = 1 / dpsidrho
Rmaj = (np.max(R) + np.min(R)) / 2


twist_shift_geo_fac = 2.*shat*gds21/gds22
jtwist = (twist_shift_geo_fac)/y0*x0

####################################################################
##########--------FIELD-LINE CUT CALCULATION----------##############
####################################################################

if flux_tube_cut == "gds21":
    print("***************************************************************************")
    print("You have chosen to cut the flux tube to enforce exact periodicity (gds21=0)")
    print("***************************************************************************")

    from scipy.interpolate import splrep, PPoly
    tck = splrep(theta, gds21, s=0)
    ppoly = PPoly.from_spline(tck)
    gds21_roots = ppoly.roots(extrapolate=False)

    if npol_min is not None:
        gds21_roots = gds21_roots[gds21_roots > npol_min*np.pi]

    # determine theta cut
    cut = gds21_roots[which_crossing]
elif flux_tube_cut == "gbdrift0":
    print("***************************************************************************************")
    print("You have chosen to cut the flux tube to enforce continuous magnetic drifts (gbdrift0=0)")
    print("***************************************************************************************")

    from scipy.interpolate import splrep, PPoly
    tck = splrep(theta, gbdrift0, s=0)
    ppoly = PPoly.from_spline(tck)
    gbdrift0_roots = ppoly.roots(extrapolate=False)

    if npol_min is not None:
        gbdrift0_roots = gbdrift0_roots[gbdrift0_roots > npol_min*np.pi]

    # determine theta cut
    cut = gbdrift0_roots[which_crossing]
elif flux_tube_cut == "aspect":
    print("*************************************************************************")
    print("You have chosen to cut the flux tube to enforce y0/x0 = ", y0/x0)
    print("*************************************************************************")

    jtwist_spl = CubicSpline(theta, jtwist)

    # find locations where jtwist_spl is integer valued. we'll check jtwist = [-30, 30] unless jtwist_max is set
    if jtwist_in is not None:
        vals = np.array([-jtwist_in, jtwist_in])
    elif jtwist_max is not None:
        vals =  np.arange(-jtwist_max,jtwist_max)
    else:
        vals =  np.arange(-30,30)
    vals = vals[(vals < -0.1) | (vals > 0.1)] # omit jtwist = 0
    crossings = [jtwist_spl.solve(i, extrapolate=False) for i in vals]
    crossings = np.concatenate(crossings)
    crossings.sort()

    if npol_min is not None:
        crossings = crossings[crossings > npol_min*np.pi]

    # determine theta cut
    cut = crossings[which_crossing]
elif flux_tube_cut == "none":
    print("***************************************************")
    print("You have chosen not to take a cut of the flux tube.")
    print("***************************************************")

if flux_tube_cut != "none":
    # new truncated theta array
    theta_cut = np.linspace(-cut, cut, ntheta)

    # interpolate geometry arrays onto new truncated theta array
    bmag_spl = InterpolatedUnivariateSpline(theta, bmag)
    bmag = bmag_spl(theta_cut)

    gradpar_spl = InterpolatedUnivariateSpline(theta, gradpar)
    gradpar = gradpar_spl(theta_cut)

    cvdrift_spl = InterpolatedUnivariateSpline(theta, cvdrift)
    cvdrift = cvdrift_spl(theta_cut)

    cvdrift0_spl = InterpolatedUnivariateSpline(theta, cvdrift0)
    cvdrift0 = cvdrift0_spl(theta_cut)
    
    gbdrift_spl = InterpolatedUnivariateSpline(theta, gbdrift)
    gbdrift = gbdrift_spl(theta_cut)

    gbdrift0_spl = InterpolatedUnivariateSpline(theta, gbdrift0)
    gbdrift0 = gbdrift0_spl(theta_cut)

    gds2_spl = InterpolatedUnivariateSpline(theta, gds2)
    gds2 = gds2_spl(theta_cut)

    gds21_spl = InterpolatedUnivariateSpline(theta, gds21)
    gds21 = gds21_spl(theta_cut)

    gds22_spl = InterpolatedUnivariateSpline(theta, gds22)
    gds22 = gds22_spl(theta_cut)

    grho_spl = InterpolatedUnivariateSpline(theta, grho)
    grho = grho_spl(theta_cut)

    R_spl = InterpolatedUnivariateSpline(theta, R)
    R = R_spl(theta_cut)

    Z_spl = InterpolatedUnivariateSpline(theta, Z)
    Z = Z_spl(theta_cut)

    theta = theta_cut

####################################################################
##########--------EQUAL-ARC THETA [-PI,PI] CALCULATION---------#####
####################################################################

theta_PEST = theta

# using 2*pi in the numerator in gradpar_eqarc scales the eq-arc theta angle to [-pi,pi]
gradpar_eqarc = 2*np.pi / (ctrap(1 / gradpar, theta, initial=0)[-1])
theta_eqarc = gradpar_eqarc * ctrap(1 / gradpar, theta, initial=0) - np.pi

domain_scaling_factor = theta[-1]/theta_eqarc[-1]

print(f"Final (unscaled) theta grid goes from [{theta[0]}, {theta[-1]}]")
print(f"domain_scaling_factor = {domain_scaling_factor} so that scaled theta grid is [-pi, pi]")

# uniformly spaced equal-arc theta grid
theta_GX = np.linspace(-np.pi, np.pi, ntheta)

# interpolate arrays onto GX theta grid
bmag_GX = np.interp(theta_GX, theta_eqarc, bmag)
gds2_GX = np.interp(theta_GX, theta_eqarc, gds2)
gds21_GX = np.interp(theta_GX, theta_eqarc, gds21)
grho_GX = np.interp(theta_GX, theta_eqarc, grho)
gds22_GX = np.interp(theta_GX, theta_eqarc, gds22)
cvdrift_GX = np.interp(theta_GX, theta_eqarc, cvdrift)
cvdrift0_GX = np.interp(theta_GX, theta_eqarc, cvdrift0)
gbdrift_GX = np.interp(theta_GX, theta_eqarc, gbdrift)
gbdrift0_GX = np.interp(theta_GX, theta_eqarc, gbdrift0)
R_GX = np.interp(theta_GX, theta_eqarc, R)
Z_GX = np.interp(theta_GX, theta_eqarc, Z)
gradpar_GX = gradpar_eqarc * np.ones((len(bmag_GX),))

#####################################################################
##############-----------GX SAVE FORMAT-------------#################
#####################################################################
try:
    # import netCDF4 as nc
    # eikfile_nc = stem + ".eiknc.nc"
    eikfile_nc = eiknc

    print("Writing eikfile in netCDF format\n")

    ds0 = ds(eikfile_nc, "w")

    z_nc = ds0.createDimension("z", ntheta)

    theta_nc = ds0.createVariable("theta", "f8", ("z",))
    theta_PEST_nc = ds0.createVariable("theta_PEST", "f8", ("z",))
    bmag_nc = ds0.createVariable("bmag", "f8", ("z",))
    gradpar_nc = ds0.createVariable("gradpar", "f8", ("z",))
    grho_nc = ds0.createVariable("grho", "f8", ("z",))
    gds2_nc = ds0.createVariable("gds2", "f8", ("z",))
    gds21_nc = ds0.createVariable("gds21", "f8", ("z",))
    gds22_nc = ds0.createVariable("gds22", "f8", ("z",))
    gbdrift_nc = ds0.createVariable("gbdrift", "f8", ("z",))
    gbdrift0_nc = ds0.createVariable("gbdrift0", "f8", ("z",))
    cvdrift_nc = ds0.createVariable("cvdrift", "f8", ("z",))
    cvdrift0_nc = ds0.createVariable("cvdrift0", "f8", ("z",))
    jacob_nc = ds0.createVariable("jacob", "f8", ("z",))
    Rplot_nc = ds0.createVariable("Rplot", "f8", ("z",))
    Zplot_nc = ds0.createVariable("Zplot", "f8", ("z",))

    drhodpsi_nc = ds0.createVariable(
        "drhodpsi",
        "f8",
    )
    kxfac_nc = ds0.createVariable(
        "kxfac",
        "f8",
    )
    Rmaj_nc = ds0.createVariable(
        "Rmaj",
        "f8",
    )
    q = ds0.createVariable(
        "q",
        "f8",
    )
    shat_nc = ds0.createVariable(
        "shat",
        "f8",
    )
    scale = ds0.createVariable(
        "scale",
        "f8",
    )
    alpha_nc = ds0.createVariable(
        "alpha",
        "f8",
    )
    zeta_center_nc = ds0.createVariable(
        "zeta_center",
        "f8",
    )
    nfp_nc = ds0.createVariable(
        "nfp",
        "i4",
    )

    theta_nc[:] = theta_GX[:]
    theta_PEST_nc[:] = theta_PEST[:]
    bmag_nc[:] = bmag_GX[:]
    gradpar_nc[:] = gradpar_GX[:]
    grho_nc[:] = grho_GX[:]
    gds2_nc[:] = gds2_GX[:]
    gds21_nc[:] = gds21_GX[:]
    gds22_nc[:] = gds22_GX[:]
    gbdrift_nc[:] = gbdrift_GX[:]
    gbdrift0_nc[:] = gbdrift0_GX[:]
    cvdrift_nc[:] = cvdrift_GX[:]
    cvdrift0_nc[:] = gbdrift0_GX[:]

    Rplot_nc[:] = R_GX[:]
    Zplot_nc[:] = Z_GX[:]

    drhodpsi_nc[0] = abs(1 / dpsidrho)
    kxfac_nc[0] = kxfac 
    Rmaj_nc[0] = (np.max(Rplot_nc) + np.min(Rplot_nc)) / 2
    q[0] = qfac
    shat_nc[0] = shat
    scale[0] = domain_scaling_factor
    alpha_nc[0] = alpha
    zeta_center_nc[0] = zeta_center
    nfp_nc[0] = nfp

    ds0.close()
except ModuleNotFoundError:
    print(
        "No netCDF4 package in your Python environment...Not saving a netCDf input file"
    )
