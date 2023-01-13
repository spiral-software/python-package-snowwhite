#! python

"""
usage: run-stepphase.py  N [ d|s [ GPU|CPU ]]
  N = cube size                       (recommend 81)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
"""


import sys
from snowwhite.stepphasesolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def usage():
    print(__doc__.strip())
    sys.exit()

try:
    N = int(sys.argv[1])
except:
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

dims = [N,N,N]

src = np.ones(dims, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

if forGPU:
    src = cp.asarray(src) 

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

p1 = StepPhaseProblem(N)
s1 = StepPhaseSolver(p1, opts)

tmp = xp.fft.rfftn(src)
amplitudes = xp.absolute(tmp)

dstP = s1.runDef(src, amplitudes)
dstC = s1.solve(src, amplitudes)

diffP = xp.max ( xp.absolute ( src - dstP ) )
diffC = xp.max ( xp.absolute ( src - dstC ) ) 
diffCP = xp.max ( xp.absolute ( dstP - dstC ) ) 

print ('Diff between src and dstP =  ' + str(diffP) )
print ('Diff between src and dstC =  ' + str(diffC) )
print ('Diff between dstC and dstP = ' + str(diffCP) )
