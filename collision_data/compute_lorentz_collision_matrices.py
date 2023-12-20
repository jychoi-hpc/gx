import scipy.special
import numpy as np
import math
from scipy import integrate
from mpi4py import MPI
import sys

import os, ctypes
from scipy import LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('libcoll.so'))
lib.nuD_proj.restype = ctypes.c_double
lib.nuD_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
lib.invU3_proj.restype = ctypes.c_double
lib.invU3_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

def nuD_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.nuD_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def invU3_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.invU3_proj, user_data)
    
    return integrate.dblquad(func, -1, 1, 0, np.inf, epsrel=1e-5)[0]

def compute_nuD_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuDMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuD_proj_C(j,k,NL)
            if rank==size-1:
                print("nuD", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuDMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuDMat")
        sys.stdout.flush()

    return nuDMat

def compute_invU3_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    invU3Mat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = invU3_proj_C(j,k,NL)
            if rank==size-1:
                print("invU3", j,k, val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, invU3Mat, root=0)
    
    if rank==0:
        print("DONE COMPUTING invU3Mat")
        sys.stdout.flush()

    return invU3Mat
       
def compute_lorentz_matrix(NL, NM):
   delta = lambda x,y: 1 if x==y else 0
   
   NLM = NL*NM
   lorentzMat = np.zeros((NLM, NLM))
   for a in range(NLM):
       for b in range(NLM):
           j = a%NL
           k = a//NL
           l = b%NL
           m = b//NL
           lorentzMat[a,b] = l*math.sqrt((m+1)*(m+2))*delta(m+2,k)*delta(l-1,j) \
                            - (l + m + 2*l*m)*delta(m,k)*delta(l,j) \
                            + (l+1)*math.sqrt(m*(m-1))*delta(m-2,k)*delta(l+1,j)
        
   return lorentzMat

from netCDF4 import Dataset
        
NL=16
NM=64
NLM=NL*NM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
perrank = NL*NM//size
    
invU3Mat = compute_invU3_matrix(NL, NM)
nuDMat = compute_nuD_matrix(NL, NM)
lorentzMat = compute_lorentz_matrix(NL, NM)

comm.Barrier()

if rank==0:
    nc = Dataset('lorentz_collision_matrices.nc', 'w')
    nc.createDimension('x', NLM)
    nc.createDimension('y', NLM)
    var = nc.createVariable('nuD', 'float32', ('x', 'y'))
    var[:,:] = nuDMat

    var = nc.createVariable('invU3', 'float32', ('x', 'y'))
    var[:,:] = invU3Mat
    
    var = nc.createVariable('lorentz', 'float32', ('x', 'y'))
    var[:,:] = lorentzMat
    
    var = nc.createVariable('nuDLorentz', 'float32', ('x', 'y'))
    var[:,:] = np.matmul(nuDMat, lorentzMat)

    var = nc.createVariable('invU3Lorentz', 'float32', ('x', 'y'))
    var[:,:] = np.matmul(invU3Mat, lorentzMat)
