#! python

from snowwhite.dftsolver import *
import numpy as np
import sys

k = SW_FORWARD

c_type = 'double'
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-fftn size [ F|I [ d|s ]]")
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

opts = { SW_OPT_REALCTYPE : c_type }
    
p1 = DftProblem(n, k)
s1 = DftSolver(p1, opts)

src = np.zeros(n, cxtype)

for i in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[i] = vr + vi * 1j
        
resP = s1.runDef(src)
resC = s1.solve(src)

diff = np.max ( np.absolute ( resC - resP ) )

print ('Diff between Python/C transforms = ' + str(diff) )