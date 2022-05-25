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
opts = { SW_OPT_REALCTYPE : c_type }

##  True or False for CUDA, HIP -- requires CuPy
genCuda = True                  ##  set as default
genHIP  = False

if len ( sys.argv ) > 3:
    if sys.argv[3] == "CUDA":
        genCuda = True
    elif sys.argv[3] == "HIP":
        genCuda = False
        genHIP = True
    elif sys.argv[3] == "CPU":
        genCuda = False

genCuda = genCuda and (cp != None)
if genCuda:
    opts[SW_OPT_CUDA] = genCuda

genHIP = genHIP and (cp != None)
if genHIP:
    opts[SW_OPT_HIP] = genHIP

xp = np
if genCuda or genHIP:
    xp = cp

print ( 'N = ' + str(N) + ' c_type = ' + c_type + ' genCuda = ' + str(genCuda) + ' genHIP = ' + str(genHIP), flush = True )

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
