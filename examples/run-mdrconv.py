from snowwhite.mdrconvsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import sys

N = 32
if len(sys.argv) > 1:
    N = int ( sys.argv[1] )

    
genCuda = True
genCuda = genCuda and (cp != None)

xp = np
if genCuda:
    xp = cp


p1 = MdrconvProblem(N)
s1 = MdrconvSolver(p1, {SW_OPT_CUDA : genCuda, SW_OPT_KEEPTEMP : False, SW_OPT_PRINTRULETREE : False})

(input_data, symbol) = s1.buildTestInput()

output_Py = s1.runDef(input_data, symbol)
output_C = s1.scale(s1.solve(input_data, symbol))

diff = np.max ( np.absolute (  output_Py - output_C ))
print ( 'Max Diff between Python/C = ' + str(diff) )