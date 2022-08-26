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

(testIn, symbol) = s1.buildTestInput()

outPy = s1.runDef(testIn, symbol)
outC  = s1.scale(s1.solve(testIn, symbol))

print("outPy[0,0,0] = ", outPy[0,0,0])
print("outC[0,0,0] = ", outC[0,0,0])

