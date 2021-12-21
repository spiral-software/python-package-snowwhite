#! python

from snowwhite.mddftsolver import *
import numpy as np
import cupy as cp
import sys

dims = [32,32,32]
dimsTuple = tuple(dims)

# True of False for CUDA
genCuda = True
opts = {SW_OPT_CUDA : genCuda}

# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD
   
p1 = MddftProblem(dims, k)
s1 = MddftSolver(p1, opts)

src = np.ones(dimsTuple, complex)
for  k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k,vr + vi * 1j)

if genCuda:    
    src = cp.asarray(src)

dstP = s1.runDef(src)
dstC = s1.solve(src)

xp = cp.get_array_module(src)

diff = xp.max ( xp.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
