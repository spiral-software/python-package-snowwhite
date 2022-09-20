#! python

from snowwhite.mdprdftsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD
# base C type, 'float' or 'double'
c_type = 'double'
ftype = np.double
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-mdprdft sz [ F|I [ d|f  [ CUDA|HIP|CPU ]]]")
    print("  sz is N or N1,N2,N3")
    sys.exit()

nnn = sys.argv[1].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = [n1,n2,n3]
dimsTuple = tuple(dims)
    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE
        
if len(sys.argv) > 3:
    if sys.argv[3] == "f":
        c_type = 'float'
        ftype = np.single
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

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }
   
p1 = MdprdftProblem(dims, k)
s1 = MdprdftSolver(p1, opts)

if k == SW_FORWARD:
    src = np.ones(tuple(dims), ftype)
    for  k in range (np.size(src)):
        vr = np.random.random()
        src.itemset(k,vr)
else:
    dims2 = dims.copy()
    z = dims2.pop()
    dims2.append(z // 2 + 1)
    src = np.ones(tuple(dims2), cxtype)
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
