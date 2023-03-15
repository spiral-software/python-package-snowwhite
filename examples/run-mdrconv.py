#! python

"""
usage: run-mdrconv.py N [ d|s [ GPU|CPU ]]
  N = cube size, N >= 4
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)                     
                                    
Three-dimensional real cyclic convolution
"""

import sys
from snowwhite.mdrconvsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import sys

def usage():
    print(__doc__.strip())
    sys.exit()

try:
    N = int(sys.argv[1])
except:
    usage()
    
if N < 4:
    usage()

c_type = 'double'
src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "s":
        c_type = 'float'
        src_type = np.single

if len ( sys.argv ) > 3:
    plat_arg = sys.argv[3]
else:
    plat_arg = 'GPU'

if plat_arg == 'GPU' and (cp != None):
    platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    forGPU = True
    xp = cp
else:
    platform = SW_CPU
    forGPU = False 
    xp = np

opts = {SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform}

xp = np
if forGPU:
    xp = cp

p1 = MdrconvProblem(N)
s1 = MdrconvSolver(p1, opts)

(testIn, symbol) = s1.buildTestInput()

dstP = s1.runDef(testIn, symbol)
dstC = s1.solve(testIn, symbol)

diff = xp.max(xp.absolute(dstC - dstP))

print('Diff between Python/C transforms = ' + str(diff))



