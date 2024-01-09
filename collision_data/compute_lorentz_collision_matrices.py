import scipy.special
import numpy as np
import math
from scipy import integrate
from mpi4py import MPI
import sys

import os, ctypes
from scipy import LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('libcoll.so'))
lib.nuDLorentz_proj.restype = ctypes.c_double
lib.nuDLorentz_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.nuDFLR_proj.restype = ctypes.c_double
lib.nuDFLR_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.invU3Lorentz_proj.restype = ctypes.c_double
lib.invU3Lorentz_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.invU3FLR_proj.restype = ctypes.c_double
lib.invU3FLR_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

def nuDLorentz_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.nuDLorentz_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def nuDFLR_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.nuDFLR_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def invU3Lorentz_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.invU3Lorentz_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def invU3FLR_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.invU3FLR_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_nuDLorentz_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuDLorentzMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuDLorentz_proj_C(j,k,NL)
            if rank==size-1:
                print("nuDLorentz", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuDLorentzMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuDLorentzMat")
        sys.stdout.flush()

    return nuDLorentzMat

def compute_nuDFLR_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuDFLRMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuDFLR_proj_C(j,k,NL)
            if rank==size-1:
                print("nuDFLR", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuDFLRMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuDFLRMat")
        sys.stdout.flush()

    return nuDFLRMat

def compute_invU3Lorentz_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    invU3LorentzMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = invU3Lorentz_proj_C(j,k,NL)
            if rank==size-1:
                print("invU3Lorentz", j,k, val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, invU3LorentzMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING invU3LorentzMat")
        sys.stdout.flush()

    return invU3LorentzMat

def compute_invU3FLR_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    invU3FLRMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = invU3FLR_proj_C(j,k,NL)
            if rank==size-1:
                print("invU3FLR", j,k, val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, invU3FLRMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING invU3FLRMat")
        sys.stdout.flush()

    return invU3FLRMat
       
from netCDF4 import Dataset
        
NL=16
NM=64
NLM=NL*NM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
perrank = NL*NM//size
    
nuDLorentzMat = compute_nuDLorentz_matrix(NL, NM)
invU3LorentzMat = compute_invU3Lorentz_matrix(NL, NM)
nuDFLRMat = compute_nuDFLR_matrix(NL, NM)
invU3FLRMat = compute_invU3FLR_matrix(NL, NM)

comm.Barrier()

if rank==0:
    nc = Dataset('lorentz_collision_matrices.nc', 'w')
    nc.createDimension('x', NLM)
    nc.createDimension('y', NLM)

    var = nc.createVariable('NL', 'i4')
    var[:] = NL

    var = nc.createVariable('NM', 'i4')
    var[:] = NM

    var = nc.createVariable('nuDLorentz', 'float32', ('x', 'y'))
    var[:,:] = nuDLorentzMat

    var = nc.createVariable('nuDFLR', 'float32', ('x', 'y'))
    var[:,:] = nuDFLRMat

    var = nc.createVariable('invU3Lorentz', 'float32', ('x', 'y'))
    var[:,:] = invU3LorentzMat

    var = nc.createVariable('invU3FLR', 'float32', ('x', 'y'))
    var[:,:] = invU3FLRMat
