#! python

from snowwhite.dftsolver import *
import numpy as np
import sys

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' xform_size dir (where dir is F/f [forward] or R/r [reverse])' )
    sys.exit ('missing argument(s)')

n = int ( sys.argv[1] )
p1 = DftProblem(n)
if sys.argv[2] == "F" or sys.argv[2] == "f":
    fwd = 1                      ##  standard python transform: 1 ==> forward, -1 reverse
else:
    fwd = -1
    
s1 = DftSolver(p1, {SW_OPT_DIRECTION : fwd})

max_diff = 0
max_diff_py = 0

src = np.zeros(n).astype(complex)

for k in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[k] = vr + vi * 1j
        
    ##  print ('Input = ' + str(src) )

    dstP = s1.runDef(src)
    if fwd == 1:
        ##  did a forward transform, do the reverse on the result
        FFT = np.fft.ifft ( dstP )
    else:
        FFT = np.fft.fft ( dstP )

    diffCP = src - FFT
    diff = np.max ( np.absolute ( diffCP ) )
    if diff > max_diff_py:
        max_diff_py = diff
        
    dstC = s1.solve(src)
        
    ##  print ('Python Output = ' + str(dstP))
    ##  print ('C Output = ' + str(dstC))

    diffCP = dstP - dstC.astype(np.double).view(complex)
    ##  print ('Difference between Python & C = ' + str(diffCP) )
    diff = np.max ( np.absolute ( diffCP ) )
    if diff > max_diff:
        max_diff = diff

if fwd == 1:
    print ('n = ' + str(n) + ': Max Diff between Python/C over all inputs: forward = ' + str(max_diff) )
else:
    print ('n = ' + str(n) + ': Max Diff between Python/C over all inputs: reverse = ' + str(max_diff) )

print ('n = ' + str(n) + ': Max Diff between Python transform/inverse = ' + str(max_diff_py) )

