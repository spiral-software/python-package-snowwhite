#! python

from snowwhite.dftsolver import *
import numpy as np
import sys

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' xform_size dir (where dir is F [forward] or I [inverse])' )
    sys.exit ('missing argument(s)')

n = int ( sys.argv[1] )
if sys.argv[2] == "F" or sys.argv[2] == "f":
    k = SW_FORWARD
else:
    k = SW_INVERSE
    
p1 = DftProblem(n, k)
s1 = DftSolver(p1)


src = np.zeros(n).astype(complex)

for i in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[i] = vr + vi * 1j
        
resP = s1.runDef(src)
resC = s1.solve(src)

diff = np.max ( np.absolute ( resC - resP ) )

print ('Diff between Python/C transforms = ' + str(diff) )