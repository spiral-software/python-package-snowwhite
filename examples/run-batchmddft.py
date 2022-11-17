#! python

from snowwhite import *
from snowwhite.batchmddftsolver import *
import numpy as np  
import sys

n = 4
b = 4

if len(sys.argv) > 1:
    n = int ( sys.argv[1] )
if len(sys.argv) > 2:    
    b = int ( sys.argv[2] )
    
p1 = BatchMddftProblem([n,n,n], b)

s1 = BatchMddftSolver(p1, {SW_OPT_PLATFORM : SW_CUDA, 
    SW_OPT_KEEPTEMP : False, SW_OPT_PRINTRULETREE : True})

input_data = s1.buildTestInput()

output_Py = s1.runDef(input_data)
output_C = s1.solve(input_data)

xp = get_array_module(output_Py)

diff = xp.max ( xp.absolute (  output_Py - output_C))
print ( 'Diff between Python/C transforms = ' + str(diff) )
