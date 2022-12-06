#! python

from snowwhite.batchmddftsolver import *
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
cxtype = np.cdouble

if (len(sys.argv) < 3) or (sys.argv[1] == '?'):
    print('run-mddft sz bat [ F|I [ d|f  [ CUDA|HIP|CPU ]]]')
    print('  sz is N or N1,N2,N3')
    print('  bat is batch size')
    sys.exit()

nnn = sys.argv[1].split(',')
n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = [n1,n2,n3]
dimsTuple = tuple(dims)

bat = int(sys.argv[2])

if len(sys.argv) > 3:
    if sys.argv[3] == 'I':
        k = SW_INVERSE
        
if len(sys.argv) > 4:
    if sys.argv[4] == 'f':
        c_type = 'float'
        cxtype = np.csingle
        
if len ( sys.argv ) > 5:
    plat_arg = sys.argv[5]
else:
    plat_arg = 'CUDA' if (cp != None) else 'CPU'
    
if plat_arg == 'CUDA' and (cp != None):
    platform = SW_CUDA
    forGPU = True
    xp = cp
elif plat_arg == 'HIP' and (cp != None):
    platform = SW_HIP
    forGPU = True
    xp = cp
else:
    platform = SW_CPU
    forGPU = False 
    xp = np       

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

p1 = BatchMddftProblem(dims, bat)
s1 = BatchMddftSolver(p1, opts)

input_data = s1.buildTestInput()

output_Py = s1.runDef(input_data)
output_C = s1.solve(input_data)

xp = get_array_module(output_Py)

diff = xp.max ( xp.absolute (  output_Py - output_C))
print ( 'Diff between Python/C transforms = ' + str(diff) )