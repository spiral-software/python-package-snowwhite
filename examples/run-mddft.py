#! python

from snowwhite.mddftsolver import *
import numpy as np
import sys

dims = [32,32,32]
dimsTuple = tuple(dims)

# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD

# set SW_OPT_CUDA to True to generate CUDA code
    
p1 = MddftProblem(dims, k)
s1 = MddftSolver(p1, {SW_OPT_CUDA : False})

src = np.ones(dimsTuple, complex)
for  k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k,vr + vi * 1j)

dstP = s1.runDef(src)
dstC = s1.solve(src)

diff = np.max ( np.absolute ( dstC - dstP ) )

print ('Diff between Python/C transforms = ' + str(diff) )
