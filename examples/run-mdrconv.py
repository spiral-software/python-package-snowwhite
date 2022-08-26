#! python

import sys
from snowwhite.mdrconvsolver import *
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

xp = np
if forGPU:
    xp = cp

p1 = MdrconvProblem(N)
s1 = MdrconvSolver(p1, opts)

for t in range(8):
    for i in range(t+1,t+9):
        shift = (i,i,i)
        target = (t,t,t)
        print('shift'+str(shift)+', target'+str(target))
        (testIn, symbol) = s1.buildTestInput(shift, target)
        outPy = s1.runDef(testIn, symbol)
        outC  = s1.scale(s1.solve(testIn, symbol))
        print('outPy %.5f' % outPy[target])
        print('outC  %.5f' % outC[target])



