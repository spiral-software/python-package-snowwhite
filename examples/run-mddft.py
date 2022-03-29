#! python

from snowwhite.mddftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

dims = [32,32,32]
dimsTuple = tuple(dims)

# True of False for CUDA, CUDA requires CuPy
genCuda = True
genCuda = genCuda and (cp != None)
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

xp = np
if genCuda:    
    src = cp.asarray(src)
    xp = cp

dstP = s1.runDef(src)
dstC = s1.solve(src)

diff = xp.max ( xp.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
