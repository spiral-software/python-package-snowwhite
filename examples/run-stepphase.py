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

xp = np
if genCuda:
    xp = cp


p1 = StepPhaseProblem(N)
s1 = StepPhaseSolver(p1, opts)

src = np.ones(dimsTuple, dtype=np.double)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

if genCuda:
    src = cp.asarray(src) 

amplitudes = xp.absolute(xp.fft.rfftn(src))

dstP = s1.runDef(src, amplitudes)
dstC = s1.solve(src, amplitudes)

diffP = xp.max ( xp.absolute ( src - dstP ) )
diffC = xp.max ( xp.absolute ( src - dstC ) ) 
diffCP = xp.max ( xp.absolute ( dstP - dstC ) ) 

print ('Diff between src and dstP =  ' + str(diffP) )
print ('Diff between src and dstC =  ' + str(diffC) )
print ('Diff between dstC and dstP = ' + str(diffCP) )
