#! python

import sys
from snowwhite.stepphasesolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

N = 81
if len(sys.argv) > 1:
    N = int ( sys.argv[1] )

c_type = 'double'
src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "f":
        c_type = 'float'
        src_type = np.single

dims = [N,N,N]
dimsTuple = tuple(dims)

##  True or False for CUDA, HIP -- requires CuPy
genCuda = True                  ##  set as default
genHIP  = False

xp = np
platform = SW_CPU
if len ( sys.argv ) > 3:
    if sys.argv[3] == "CUDA" and (cp != None):
        platform = SW_CUDA
        xp = cp
    elif sys.argv[3] == "HIP" and (cp != None):
        platform = SW_HIP
        xp = cp

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

src = np.ones(dimsTuple, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

if genCuda or genHIP:
    src = cp.asarray(src) 

tmp = xp.fft.rfftn(src)
amplitudes = xp.absolute ( tmp )  ##  xp.fft.rfftn(src))

p1 = StepPhaseProblem(N)
s1 = StepPhaseSolver(p1, opts)

dstP = s1.runDef(src, amplitudes)
dstC = s1.solve(src, amplitudes)

diffP = xp.max ( xp.absolute ( src - dstP ) )
diffC = xp.max ( xp.absolute ( src - dstC ) ) 
diffCP = xp.max ( xp.absolute ( dstP - dstC ) ) 

print ('Diff between src and dstP =  ' + str(diffP) )
print ('Diff between src and dstC =  ' + str(diffC) )
print ('Diff between dstC and dstP = ' + str(diffCP) )
