#! python

import sys
from snowwhite.mddftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

N = 32
if len(sys.argv) > 1:
    N = int ( sys.argv[1] )

c_type = 'double'
src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "f":
        c_type = 'float'
        src_type = np.single

if len ( sys.argv ) > 3:
    plat_arg = sys.argv[3]
else:
    plat_arg = "CUDA"
    
if plat_arg == "CUDA" and (cp != None):
    platform = SW_CUDA
    forGPU = True
    xp = cp
elif plat_arg == "HIP" and (cp != None):
    platform = SW_HIP
    forGPU = True
    xp = cp
else:
    platform = SW_CPU
    forGPU = False 
    xp = np

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

dims = [N,N,N]        
dimsTuple = tuple(dims)

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
if forGPU:    
    src = cp.asarray(src)
    xp = cp

dstP = s1.runDef(src)
dstC = s1.solve(src)

diff = xp.max ( xp.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
