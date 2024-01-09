import scipy.special
import numpy as np
from math import sqrt
from scipy import integrate
from mpi4py import MPI
import sys

import os, ctypes
from scipy import LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('libcoll.so'))
lib.nuParDiffT0_proj.restype = ctypes.c_double
lib.nuParDiffT0_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.nuParFLR_proj.restype = ctypes.c_double
lib.nuParFLR_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

def nuParDiffT0_proj_C(a,b, NL):
    j = a%NL
    k = a//NL
    l = b%NL
    m = b//NL

    data = (ctypes.c_int*4)()
    data[0] = j
    data[1] = k
    data[2] = l
    data[3] = m
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.nuParDiffT0_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def nuParFLR_proj_C(a,b, NL):
    j = a%NL
    k = a//NL
    l = b%NL
    m = b//NL

    data = (ctypes.c_int*4)()
    data[0] = j
    data[1] = k
    data[2] = l
    data[3] = m
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.nuParFLR_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_nuParDiffT0_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuParDiffT0Mat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuParDiffT0_proj_C(j,k,NL)
            if rank==size-1:
                print("nuParDiffT0", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuParDiffT0Mat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuParDiffT0Mat")
        sys.stdout.flush()

    return nuParDiffT0Mat

def compute_nuParFLR_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuParFLRMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuParFLR_proj_C(j,k,NL)
            if rank==size-1:
                print("nuParFLR", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuParFLRMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuParFLRMat")
        sys.stdout.flush()

    return nuParFLRMat

from netCDF4 import Dataset
        
NL=16
NM=64
NLM=NL*NM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
perrank = NL*NM//size
    
nuParDiffT0Mat = compute_nuParDiffT0_matrix(NL, NM)
nuParFLRMat = compute_nuParFLR_matrix(NL, NM)

comm.Barrier()

if rank==0:
    nc = Dataset('diffusion_collision_matrices.nc', 'w')
    nc.createDimension('x', NLM)
    nc.createDimension('y', NLM)

    var = nc.createVariable('NL', 'i4')
    var[:] = NL

    var = nc.createVariable('NM', 'i4')
    var[:] = NM

    var = nc.createVariable('nuParDiffT0', 'float32', ('x', 'y'))
    var[:,:] = nuParDiffT0Mat

    var = nc.createVariable('nuParFLR', 'float32', ('x', 'y'))
    var[:,:] = nuParFLRMat
