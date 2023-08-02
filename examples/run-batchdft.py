
import argparse
import snowwhite as sw
from snowwhite.dftsolver import *
import re
import numpy as np
import sys

try:
    def parse_stride(value):
        return 'AVec' if re.match ( 'avec', value, re.IGNORECASE ) else 'APar'

    def parse_bool(value):
        return re.match ( 'true', value, re.IGNORECASE ) is not None

    def parse_type(value):
        return 'float' if re.match ( 'float', value, re.IGNORECASE ) else 'double'

    def parse_platform(value):
        return 'GPU' if re.match ( 'gpu', value, re.IGNORECASE ) else 'CPU'

    parser = argparse.ArgumentParser ( description = 'Validate FFTX built batch 1D FFTs against NumPy computed versions of the transforms' )

    ##  Positional argument:: batch, DFT_leng
    parser.add_argument ( 'batch', type=str, help='2D batch size, expressed as "KxM"' )
    parser.add_argument ( 'DFT_leng', type=int, help='DFT length, N' )

    ##  Optional arguments (all are case insensitive):
    ##  All optional arguments have defaults but if you need to override an argument, all
    ##  prior args must be passed as they are positionally significant (to avoid specifying an
    ##  arg type with each): rd_stride, wr_stride, fwd_or_inv, data_type

    parser.add_argument ( 'rd_stride', type=parse_stride, nargs='?', default='APar', choices=['APar', 'AVec'],
                          help='read stride value: either APar [default] or AVec' )
    parser.add_argument ( 'wr_stride', type=parse_stride, nargs='?', default='APar', choices=['APar', 'AVec'],
                          help='write stride value: either APar [default] or AVec' )
    parser.add_argument ( 'fwd_or_inv', type=parse_bool, nargs='?', default=True,
                          help='True [default, forward transform] | False' )
    parser.add_argument ( 'data_type', type=parse_type, nargs='?', default='double', choices=['double', 'float'],
                          help='Data type: double [default] | float' )
    parser.add_argument ( 'plat_arg',  type =parse_platform, nargs='?', default='GPU', choices=['GPU', 'CPU'],
                          help='Platform: GPU [system determines CUDA or HIP, default] | CPU' )

    args = parser.parse_args()

    _bdims = re.split ( 'x', args.batch )
    b1, b2 = int ( _bdims[0] ), int ( _bdims[1] )

    R = 1 if args.rd_stride == 'APar' else 2
    W = 1 if args.wr_stride == 'APar' else 2
    k = SW_FORWARD if args.fwd_or_inv else SW_INVERSE

    c_type = args.data_type
    cxtype = np.cfloat if c_type == 'float' else np.cdouble
    ##  cxtype = np.csingle             ## should iit be np.csingle or np.cfloat ??

    # print ( 'Batch size: {}, dims = {} x {}'.format(args.batch, b1, b2) )
    # print ( 'DFT Length: ', args.DFT_leng )
    # print ( 'Read stride: {}, R = {}'.format(args.rd_stride, R) )
    # print ( 'Write stride: {}, W = {}'.format(args.wr_stride, W) )
    # print ( 'Fwd or Inv: {}, k = {}'.format(args.fwd_or_inv, k) )
    # print ( 'Data type: ', args.data_type )
        
    plat_arg = 'GPU'
    if plat_arg == 'GPU' and cp is not None:
        platform = SW_HIP if sw.has_ROCm() else SW_CUDA
        forGPU = True
        xp = cp
    else:
        platform = SW_CPU
        forGPU = False
        xp = np

    opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

    p1 = DftProblem ( args.DFT_leng, k, batchDims=[b1, b2], readStride=R, writeStride=W )
    s1 = DftSolver ( p1, opts )

    dims = (b1, b2, args.DFT_leng) if R == 1 else (args.DFT_leng, b1, b2)

    src = np.ones(dims, cxtype)

    use_ones = False
    if use_ones:
        if R == 1:
            for i in range (dims[0]):
                for j in range (dims[1]):
                    src[i,j,:] = (i*b1) + (j+1)
        else:
            for i in range (dims[1]):
                for j in range (dims[2]):
                    src[:,i,j] = (i*b1) + (j+1)
    else:
        for i in range (dims[0]):
            for j in range (dims[1]):
                for k in range(dims[2]):
                    vr = np.random.random()
                    vi = np.random.random()
                    src[i,j,k] = vr + vi * 1j

    if forGPU:
        src = cp.asarray(src)
        xp = cp

    resP = s1.runDef(src)
    resC = s1.solve(src)

    diff = xp.max ( np.absolute ( resC - resP ) )

    if diff > 1e-7:
        print ('Python/C transforms are NOT equivalent, diff = ' + str(diff) )
    else:
        print ('Python/C transforms are equivalent, diff = ' + str(diff) )

except Exception as e:
    print('An error occurred:', str(e))
    sys.exit(1)

