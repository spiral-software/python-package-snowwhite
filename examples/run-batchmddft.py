#! python

"""
usage: run-batchmddft.py sz bat [ F|I [ d|s [ GPU|CPU ]]]
  sz is N or N1,N2,N3, all N >= 2
  bat is 1D batch size, >= 1
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
  
1D batch of 3D complex FFTs
"""

import snowwhite as sw
from snowwhite.batchmddftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

def usage():
    print(__doc__.strip())
    sys.exit()

# array dimensions and batch size
try:
  nnn = sys.argv[1].split(',')
  n1 = int(nnn[0])
  n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
  n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()
  dims = [n1,n2,n3]

  bat = int(sys.argv[2])
except:
  usage()
  
if any(n < 2 for n in dims) or bat < 1:
    usage()
  
# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD
if len(sys.argv) > 3:
    if sys.argv[3] == 'I':
        k = SW_INVERSE

# base C type, 'float' or 'double'
c_type = 'double'
cxtype = np.cdouble       
if len(sys.argv) > 4:
    if sys.argv[4] == 's':
        c_type = 'float'
        cxtype = np.csingle

if len ( sys.argv ) > 5:
    plat_arg = sys.argv[5]
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

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

p1 = BatchMddftProblem(dims, bat, k)
s1 = BatchMddftSolver(p1, opts)

input_data = s1.buildTestInput()

output_Py = s1.runDef(input_data)
output_C = s1.solve(input_data)

xp = get_array_module(output_Py)

diff = xp.max ( xp.absolute (  output_Py - output_C))
print ( 'Diff between Python/C transforms = ' + str(diff) )