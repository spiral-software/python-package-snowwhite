#! python

from snowwhite.mddftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

# default cube dimension
N = 32
# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD
# base C type, 'float' or 'double'
c_type = 'double'
cxtype = np.cdouble


if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-mddft sz [ F|I [ d|f  [ CUDA|HIP|CPU ]]]")
    sys.exit()

N = int ( sys.argv[1] )
    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE
        
if len(sys.argv) > 3:
    if sys.argv[3] == "f":
        c_type = 'float'
        cxtype = np.csingle
        
if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "CUDA" if (cp != None) else "CPU"
    
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

dims = [N,N,N]
dimsTuple = tuple(dims)


opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

   
p1 = MddftProblem(dims, k)
s1 = MddftSolver(p1, opts)

src = np.ones(dimsTuple, cxtype)
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
