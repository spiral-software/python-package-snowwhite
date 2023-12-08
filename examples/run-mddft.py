#! python

"""
usage: run-mddft.py sz [ F|I [ d|s [ GPU|CPU [Fortran]]]
  sz is N or N1,N2,.. all N >= 2, single N implies 3D cube
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
  GPU is default target unless none exists or no CuPy
  C ordering is default unless Fortran specified
  
Multi-dimensional complex FFT
"""

from snowwhite.mddftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

def usage():
    print(__doc__.strip())
    sys.exit()
   
# array dimensions
try:
  nnn = sys.argv[1].split(',')
  n1 = int(nnn[0])
  if len(nnn) < 2:
    # default to 3D cube
    dims = [n1,n1,n1]
  else:
    dims = [n1]
    for i in range(1, len(nnn)):
      dims.append(int(nnn[i]))
except:
  usage()
  
if any(n < 2 for n in dims):
    usage()

# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE
 
# base C type, 'float' or 'double'
c_type = 'double'
cxtype = np.cdouble       
if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        c_type = 'float'
        cxtype = np.csingle
        
if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = 'GPU'
    
order = 'C'
if (len ( sys.argv ) > 5) and (sys.argv[5].lower() == 'fortran'):
    order = 'F'


if plat_arg == 'GPU' and (cp != None):
    platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    forGPU = True
    xp = cp
else:
    platform = SW_CPU
    forGPU = False 
    xp = np       

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }
if order == 'F':
    opts[SW_OPT_COLMAJOR] = True
   
p1 = MddftProblem(dims, k)
s1 = MddftSolver(p1, opts)

src = np.ones(dims, cxtype, order)
for  x in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(x,vr + vi * 1j)

xp = np
if forGPU:    
    src = cp.asarray(src)
    xp = cp

dstP = s1.runDef(src)
dstC = s1.solve(src)

diff = xp.max ( xp.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
