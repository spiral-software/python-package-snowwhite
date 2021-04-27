#! python

from snowwhite.batchmddftsolver import *
import numpy as np
import sys

n = 4
b = 4

if len(sys.argv) > 1:
    n = int ( sys.argv[1] )
if len(sys.argv) > 2:    
    b = int ( sys.argv[2] )
    
p1 = BatchMddftProblem(n, b)

s1 = BatchMddftSolver(p1, {SW_OPT_CUDA : True, 
    SW_OPT_KEEPTEMP : False, SW_OPT_PRINTRULETREE : True})

input_data_Py, input_data_C = s1.buildTestInput()

print('calling Python func \n')
output_Py = s1.runDef(input_data_Py)
# complex converted to interleaved for comparison with output_C
output_Py_IL = output_Py.view(dtype=np.double).flatten()

print('calling C func \n')
output_C = s1.solve(input_data_C)

print('length Py output: ', len(output_Py.flatten()), '| datatype: ', output_Py.dtype)
print('length C output: ', len(output_C), '| datatype: ', output_C.dtype, '\n')

print('calling np.max')
diff = np.max ( np.absolute (  output_Py_IL - output_C))
print ( 'Max Diff between Python/C = ' + str(diff) )