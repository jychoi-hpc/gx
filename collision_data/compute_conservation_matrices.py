import scipy.special
import numpy as np
from math import sqrt
from scipy import integrate
from mpi4py import MPI
import sys

import os, ctypes
from scipy import LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('libcoll.so'))
lib.alphaE_proj.restype = ctypes.c_double
lib.alphaE_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.alphaParL_proj.restype = ctypes.c_double
lib.alphaParL_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.alphaPerpL_proj.restype = ctypes.c_double
lib.alphaPerpL_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.alphaParD_proj.restype = ctypes.c_double
lib.alphaParD_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.alphaPerpD_proj.restype = ctypes.c_double
lib.alphaPerpD_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

def alphaE_proj_C(l, m, n):
    data = (ctypes.c_int*3)()
    data[0] = l
    data[1] = m
    data[2] = n
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.alphaE_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_alphaE_matrix(NL, NM):
    sys.stdout.flush()
    
    mat = np.zeros((perrank, NM, NL))
    alphaE = np.zeros((NL, NM, NL))

    for l in range(rank*perrank, (rank+1)*perrank):
       for m in range(NM):
          for n in range(NL):
             val = alphaE_proj_C(l, m, n)
             if rank == size-1:
                print("alphaE", l, m, n, val)
                sys.stdout.flush()
             mat[l-rank*perrank,m,n] = val

    comm.Gather(mat, alphaE, root=0)
    
    return alphaE

def alphaParL_proj_C(l, m, n):
    data = (ctypes.c_int*3)()
    data[0] = l
    data[1] = m
    data[2] = n
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.alphaParL_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_alphaParL_matrix(NL, NM):
    sys.stdout.flush()
    
    mat = np.zeros((perrank, NM, NL))
    alphaParL = np.zeros((NL, NM, NL))

    for l in range(rank*perrank, (rank+1)*perrank):
       for m in range(NM):
          for n in range(NL):
             val = alphaParL_proj_C(l, m, n)
             if rank == size-1:
                print("alphaParL", l, m, n, val)
                sys.stdout.flush()
             mat[l-rank*perrank,m,n] = val

    comm.Gather(mat, alphaParL, root=0)
    
    return alphaParL

def alphaPerpL_proj_C(l, m, n):
    data = (ctypes.c_int*3)()
    data[0] = l
    data[1] = m
    data[2] = n
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.alphaPerpL_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_alphaPerpL_matrix(NL, NM):
    sys.stdout.flush()
    
    mat = np.zeros((perrank, NM, NL))
    alphaPerpL = np.zeros((NL, NM, NL))

    for l in range(rank*perrank, (rank+1)*perrank):
       for m in range(NM):
          for n in range(NL):
             val = alphaPerpL_proj_C(l, m, n)
             if rank == size-1:
                print("alphaPerpL", l, m, n, val)
                sys.stdout.flush()
             mat[l-rank*perrank,m,n] = val

    comm.Gather(mat, alphaPerpL, root=0)
    
    return alphaPerpL

def alphaParD_proj_C(l, m, n):
    data = (ctypes.c_int*3)()
    data[0] = l
    data[1] = m
    data[2] = n
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.alphaParD_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_alphaParD_matrix(NL, NM):
    sys.stdout.flush()
    
    mat = np.zeros((perrank, NM, NL))
    alphaParD = np.zeros((NL, NM, NL))

    for l in range(rank*perrank, (rank+1)*perrank):
       for m in range(NM):
          for n in range(NL):
             val = alphaParD_proj_C(l, m, n)
             if rank == size-1:
                print("alphaParD", l, m, n, val)
                sys.stdout.flush()
             mat[l-rank*perrank,m,n] = val

    comm.Gather(mat, alphaParD, root=0)
    
    return alphaParD

def alphaPerpD_proj_C(l, m, n):
    data = (ctypes.c_int*3)()
    data[0] = l
    data[1] = m
    data[2] = n
    user_data = ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    
    func = LowLevelCallable(lib.alphaPerpD_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_alphaPerpD_matrix(NL, NM):
    sys.stdout.flush()
    
    mat = np.zeros((perrank, NM, NL))
    alphaPerpD = np.zeros((NL, NM, NL))

    for l in range(rank*perrank, (rank+1)*perrank):
       for m in range(NM):
          for n in range(NL):
             val = alphaPerpD_proj_C(l, m, n)
             if rank == size-1:
                print("alphaPerpD", l, m, n, val)
                sys.stdout.flush()
             mat[l-rank*perrank,m,n] = val

    comm.Gather(mat, alphaPerpD, root=0)
    
    return alphaPerpD

from netCDF4 import Dataset
        
NL=16
NM=64
NLM=NL*NM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
perrank = NL//size
    
alphaE = compute_alphaE_matrix(NL, NM)
alphaParL = compute_alphaParL_matrix(NL, NM)
alphaParD = compute_alphaParD_matrix(NL, NM)
alphaPerpL = compute_alphaPerpL_matrix(NL, NM)
alphaPerpD = compute_alphaPerpD_matrix(NL, NM)

comm.Barrier()

if rank==0:
    nc = Dataset('conservation_matrices.nc', 'w')
    nc.createDimension('l', NL)
    nc.createDimension('m', NM)
    nc.createDimension('n', NL)

    var = nc.createVariable('NL', 'i4')
    var[:] = NL

    var = nc.createVariable('NM', 'i4')
    var[:] = NM

    var = nc.createVariable('alphaE', 'float32', ('l', 'm', 'n'))
    var[:,:,:] = alphaE

    var = nc.createVariable('alphaParL', 'float32', ('l', 'm', 'n'))
    var[:,:,:] = alphaParL

    var = nc.createVariable('alphaParD', 'float32', ('l', 'm', 'n'))
    var[:,:,:] = alphaParD

    var = nc.createVariable('alphaPerpL', 'float32', ('l', 'm', 'n'))
    var[:,:,:] = alphaPerpL

    var = nc.createVariable('alphaPerpD', 'float32', ('l', 'm', 'n'))
    var[:,:,:] = alphaPerpD




