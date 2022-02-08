#! python

from snowwhite.stepphasesolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

N = 81
dims = [N,N,N]
dimsTuple = tuple(dims)

# True of False for CUDA, CUDA requires CuPy
genCuda = True
genCuda = genCuda and (cp != None)
opts = {SW_OPT_CUDA : genCuda}

p1 = StepPhaseProblem(N)
s1 = StepPhaseSolver(p1, opts)

src = np.ones(dimsTuple, dtype=complex)
amplitudes = np.ones(dimsTuple, dtype=np.float64)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random() + j*np.random.random())
    amplitudes.itemset(k,np.random.random())

xp = np
if genCuda:    
    src = cp.asarray(src)    
    amplitudes = cp.asarray(amplitudes)
    xp = cp

dstP = s1.runDef(src, amplitudes)
dstC = s1.solve(src, amplitudes)

diff = xp.max ( xp.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
