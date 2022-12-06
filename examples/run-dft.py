#! python

from snowwhite.dftsolver import *
import numpy as np
import sys

k = SW_FORWARD

c_type = 'double'
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-dft size [ F|I [ d|s ]]")
    print("  F  = Forward, I = Inverse")
    print("  d  = double, s = single precision")
    sys.exit()

n = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE

if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        c_type = 'float'
        cxtype = np.csingle

if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "CPU"
    
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
    
p1 = DftProblem(n, k)
s1 = DftSolver(p1, opts)

src = np.zeros(n, cxtype)

for i in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[i] = vr + vi * 1j
    
xp = np
if forGPU:    
    src = cp.asarray(src)
    xp = cp    
        
resP = s1.runDef(src)
resC = s1.solve(src)

diff = xp.max ( np.absolute ( resC - resP ) )

print ('Diff between Python/C transforms = ' + str(diff) )