import scipy.special
import numpy as np
from math import sqrt
from scipy import integrate
#from mpi4py import MPI
import sys

import os, ctypes
from scipy import LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('libcoll.so'))
lib.nuPar_proj.restype = ctypes.c_double
lib.nuPar_proj.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

def nuPar_proj_C(a,b, NL):
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
    
    func = LowLevelCallable(lib.nuPar_proj, user_data)
    
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

def compute_nuPar_matrix(NL, NM):
    NLM = NL*NM
    
    sys.stdout.flush()
    
    mat = np.zeros((perrank,NLM))
    nuParMat = np.zeros((NLM,NLM))
    
    for j in range(rank*perrank, (rank+1)*perrank):
        for k in range(NLM):
            val = nuPar_proj_C(j,k,NL)
            if rank==size-1:
                print("nuPar", j,k,val)
                sys.stdout.flush()
            mat[j-rank*perrank,k] = val
    
    comm.Gather(mat, nuParMat, root=0)
    
    if rank==0:
        print("DONE COMPUTING nuParMat")
        sys.stdout.flush()

    return nuParMat

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

def compute_DiffT0_matrix(NL, NM):
   delta = lambda x,y: 1 if x==y else 0
   
   NLM = NL*NM
   diffMat = np.zeros((NLM, NLM))
   for a in range(NLM):
       for b in range(NLM):
           l = a%NL
           m = a//NL
           j = b%NL
           k = b//NL
           diffMat[a,b] = 2*(l-1)*delta(l-2,j)*delta(m,k) + l*sqrt(m*(m-1))*delta(l-1,j)*delta(m-2,k) \
                         + l*(m+4*l+3)*delta(l-1,j)*delta(m,k) + 0.5*sqrt((m-3)*(m-2)*(m-1)*(m))*delta(l,j)*delta(m-4,k) \
                         + sqrt(m*(m-1))*(m+l+1)*delta(l,j)*delta(m-2,k) + 0.5*(7 + 4*l*l + m*(m+4) + 2*l*(m+5))*delta(l,j)*delta(m,k)
   return diffMat

# s/H\[\(.\{-}\), \(.\{-}\)]/delta(\1,j)*delta(\2,k)/g
# s/sqrt(\(.\{-}\))\*sqrt(\(.\{-}\))/sqrt((\1)*(\2))/g
def compute_DiffT_matrix(NL, NM):
   delta = lambda x,y: 1 if x==y else 0
   
   NLM = NL*NM
   diffMat = np.zeros((NLM, NLM))
   for a in range(NLM):
       for b in range(NLM):
           l = a%NL
           m = a//NL
           j = b%NL
           k = b//NL
           diffMat[a,b] = -2*(-1 + l)*l*delta(-2 + l,j)*delta(m,k) - (3*l*sqrt((-1 + m)*(m))*delta(-1 + l,j)*delta(-2 + m,k))/2 \
                          - (l*(3 + 12*l + 5*m)*delta(-1 + l,j)*delta(m,k))/2 - l*sqrt((1 + m)*(2 + m))*delta(-1 + l,j)*delta(2 + m,k) \
                          - (sqrt(((-3 + m)*(-2 + m))*((-1 + m))*(m))*delta(l,j)*delta(-4 + m,k))/2 \
                          - (sqrt((-1 + m)*(m))*(1 + 4*l + 3*m)*delta(l,j)*delta(-2 + m,k))/2 \
                          - ((8 + 12*l*l + 7*m + 3*m*m + 2*l*(8 + 3*m))*delta(l,j)*delta(m,k))/2 \
                          - (sqrt((1 + m)*(2 + m))*(3 + 2*l + m)*delta(l,j)*delta(2 + m,k))/2 \
                          - ((1 + l)*sqrt((-1 + m)*(m))*delta(1 + l,j)*delta(-2 + m,k))/2 \
                          - ((1 + l)*(5 + 4*l + m)*delta(1 + l,j)*delta(m,k))/2

   return diffMat

from netCDF4 import Dataset
        
NL=16
NM=64
NLM=NL*NM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
perrank = NL*NM//size
    
nuParMat = compute_nuPar_matrix(NL, NM)
nuParFLRMat = compute_nuParFLR_matrix(NL, NM)
DiffT0Mat = compute_DiffT0_matrix(NL, NM)
DiffTMat = compute_DiffT_matrix(NL, NM)

comm.Barrier()

if rank==0:
    nc = Dataset('diffusion_collision_matrices.nc', 'w')
    nc.createDimension('x', NLM)
    nc.createDimension('y', NLM)
    var = nc.createVariable('nuPar', 'float32', ('x', 'y'))
    var[:,:] = nuParMat

    var = nc.createVariable('nuParFLR', 'float32', ('x', 'y'))
    var[:,:] = nuParFLRMat
    
    var = nc.createVariable('nuParDiffT0', 'float32', ('x', 'y'))
    var[:,:] = np.matmul(nuParMat, DiffT0Mat)

    var = nc.createVariable('nuParDiffT', 'float32', ('x', 'y'))
    var[:,:] = np.matmul(nuParMat, DiffTMat)

