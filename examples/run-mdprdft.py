#! python

from snowwhite.mdprdftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

# default cube dimenstion
N = 48
# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD

if len(sys.argv) > 1:
    N = int ( sys.argv[1] )
    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE

dims = [N,N,N]

# True of False for CUDA, CUDA requires CuPy
genCuda = True
genCuda = genCuda and (cp != None)
opts = {SW_OPT_CUDA : genCuda}


p1 = MdprdftProblem(dims, k)
s1 = MdprdftSolver(p1, opts)

if k == SW_FORWARD:
    src = np.ones(tuple(dims), np.double)
    for  k in range (np.size(src)):
        vr = np.random.random()
        src.itemset(k,vr)
else:
    dims2 = dims.copy()
    z = dims2.pop()
    dims2.append(z // 2 + 1)
    src = np.ones(tuple(dims2), complex)
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
