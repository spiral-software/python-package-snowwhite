
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import ctypes
import sys
import random

class PrdftProblem(SWProblem):
    """Define 1D PRDFT problem."""

    def __init__(self, n, k=SW_FORWARD, batchDims=[1,1], readStride=1, writeStride=1):
        """Setup problem specifics for 1D PRDFT solver.
        
        Arguments:
        k           -- direction
        n           -- dimension of 1D PRDFT
        batchDims   -- dimensions of batch
        readStride  -- unit (1) or block (!=1)
        writeStride -- unit (1) or block (!=1)
        """
        super(PrdftProblem, self).__init__([n], k)
        self._batchDims = batchDims
        self._readStride = readStride
        self._writeStride = writeStride


class PrdftSolver(SWSolver):
    def __init__(self, problem: PrdftProblem, opts = {}):
        if not isinstance(problem, PrdftProblem):
            raise TypeError("problem must be a PrdftProblem")
        
        typ = 'd'
        self._ftype = np.double
        self._cxtype = np.cdouble
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'f'
            self._ftype = np.single
            self._cxtype = np.csingle
        n = str(problem.dimN())
        c = '_'
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = typ + 'prdft' + c + n
        else:
            namebase = typ + 'ipidft' + c + n
            
        dims = problem._batchDims
        bat = np.prod(dims)
        if bat > 1:
            namebase = namebase + c + 'b' + 'x'.join([str(n) for n in dims])
            namebase = namebase + ('p' if problem._writeStride == 1 else 'v')
            namebase = namebase + ('p' if problem._readStride == 1 else 'v')
            
        opts[SW_OPT_METADATA] = True
            
        super(PrdftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        xp = get_array_module(src)

        ax = -1 if self._problem._readStride == 1 else 0
        
        if self._problem.direction() == SW_FORWARD:
            dst = xp.fft.rfft(src, axis=ax)
        else:
            dst = xp.fft.irfft(src, n=self._problem.dimN(), axis=ax)
            
        if self._problem._writeStride != self._problem._readStride:
            if self._problem._writeStride != 1:
                # Par Vec
                # new shape is inverse of src
                revdims = src.shape[::-1]
                drt = dst.reshape(np.asarray(revdims[1:]).prod(), revdims[0]).transpose()
                dst = drt.reshape(revdims)
            else:
                # Vec Par
                dims = dst.shape
                drt = dst.reshape(dims[0], np.asarray(dims[1:]).prod()).transpose()
                dst = drt.reshape(dims[::-1])

        return dst
        
    def _trace(self):
        pass
        
    def _new_dst(self, src):
        xp = get_array_module(src)
        Nx = self._problem.dimN()
        typ = self._ftype
        if self._problem.direction() == SW_FORWARD:
            Nx = (Nx // 2) + 1
            typ = self._cxtype
        bdims = self._problem._batchDims
        if self._problem._writeStride == 1:
            dims = bdims + [Nx]
        else:
            dims = [Nx] + bdims
        dst = xp.zeros(dims, typ)
        return dst

    def solve(self, src, dst=None):
        """Call SPIRAL-generated function."""
        if type(dst) == type(None):
            dst = self._new_dst(src)
        self._func(dst, src)
        return dst

    def _writeScript(self, script_file):
        filename = self._namebase
        nameroot = self._namebase
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        if self._genHIP:
            filetype = '.cpp'
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file)
        print("Import(realdft);", file = script_file)

        print('', file = script_file)
        
        if self._problem.direction() == SW_FORWARD:
            xform = 'PRDFT'
        else:
            xform = 'IPRDFT'
        
        dft_def = xform + '(N, -1)'
        #if self._problem.direction() == SW_INVERSE:
        #    dft_def = 'Scale(1/N, ' + dft_def + ')'
        
        bdims = self._problem._batchDims
        bdims_str = str(np.prod(bdims))
        
        W = 'APar' if self._problem._writeStride == 1 else 'AVec'
        R = 'APar' if self._problem._readStride == 1 else 'AVec'
        
        print('t := let(', file = script_file) 
        print('    name := "' + nameroot + '",', file = script_file)
        print('    N  := ' + str(self._problem.dimN()) + ',', file = script_file)
        print('    TFCall(TRC(TTensorI(' + dft_def + ', ' + bdims_str + ' ,' + W + ', ' + R +')), rec(fname := name, params := []))', file = script_file)
        print(');', file = script_file)
        
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file) 
        elif self._genHIP:
            print ( 'conf := FFTXGlobals.defaultHIPConf();', file = script_file )
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file) 

        print("opts := conf.getOpts(t);", file = script_file)
        if self._genCuda or self._genHIP:
            print('opts.wrapCFuncs := true;', file = script_file)

        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)

        self._writePrintOpts(script_file)

        print('Add(opts.includes, "<float.h>");',  file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
    
    def _setFunctionMetadata(self, obj):
        bdims = self._problem._batchDims
        if np.prod(bdims) > 1:
            obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_BATPRDFT
            obj[SW_KEY_BATCHSIZE] = int(np.prod(bdims))
            obj[SW_KEY_READSTRIDE]  = SW_STR_UNIT if self._problem._readStride  == 1 else SW_STR_BLOCK
            obj[SW_KEY_WRITESTRIDE] = SW_STR_UNIT if self._problem._writeStride == 1 else SW_STR_BLOCK
        else:
            obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_PRDFT



