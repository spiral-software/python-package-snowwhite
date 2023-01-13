
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import ctypes
import sys


class StepPhaseProblem(SWProblem):
    """Define dimensions of StepPhase problem."""

    def __init__(self, n):
        """Setup problem specifics for StepPhase solver.
        
        Arguments:
        n     -- size of StepPhase cube
        """
        super(StepPhaseProblem, self).__init__([n,n,n])
        
    def dimN(self):
        return self.dimensions()[0]
        

class StepPhaseSolver(SWSolver):
    def __init__(self, problem: StepPhaseProblem, opts = {}):
        if not isinstance(problem, StepPhaseProblem):
            raise TypeError("problem must be a StepPhaseProblem")
        
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
            self._ctype  = 'float'
        else:
            typ = 'z'
            self._ctype = 'double'
             
        namebase = typ + 'stepphase_' + str(problem.dimN())

        super(StepPhaseSolver, self).__init__(problem, namebase, opts)

    def runDef(self, rho, amplitudes):
        """Solve using internal Python definition."""

        xp = get_array_module(rho)
        
        n = self._problem.dimN()
        amp_mask = xp.ones(amplitudes.shape, dtype=xp.bool_)
        amp_mask[0, 0, 0] = 0

        rho_hat = xp.fft.rfftn(rho)
        phases = xp.angle(rho_hat)
        rho_hat_mod = xp.where(
            amp_mask,
            amplitudes * xp.exp(1j*phases),
            rho_hat)

        return xp.fft.irfftn(rho_hat_mod, rho.shape)
        
    def _trace(self):
        pass

    def solve(self, src, amplitudes, dst=None):
        """Call SPIRAL-generated function."""
        
        xp = get_array_module(src)
        
        if type(dst) == type(None):
            n = self._problem.dimN()  
            dst = xp.zeros((n, n, n), src.dtype)
        self._func(dst, src, amplitudes)
        return dst
                    
    def _func(self, dst, src, amplitudes):
        """Call the SPIRAL generated main function"""

        xp = sw.get_array_module(src)
        
        if xp == np: 
            if self._genCuda or self._genHIP:
                raise RuntimeError('GPU function requires CuPy arrays')
            # NumPy array on CPU
            return self._MainFunc( 
                    dst.ctypes.data_as(ctypes.c_void_p),
                    src.ctypes.data_as(ctypes.c_void_p),
                    amplitudes.ctypes.data_as(ctypes.c_void_p))
        else:
            if not self._genCuda and not self._genHIP:
                raise RuntimeError('CPU function requires NumPy arrays')
            # CuPy array on GPU
            return self._MainFunc(
                    ctypes.cast(dst.data.ptr, ctypes.POINTER(ctypes.c_void_p)), 
                    ctypes.cast(src.data.ptr, ctypes.POINTER(ctypes.c_void_p)),
                    ctypes.cast(amplitudes.data.ptr, ctypes.POINTER(ctypes.c_void_p)))

    def _writeScript(self, script_file):
        filename = self._namebase
        nameroot = self._namebase
        ns = str(self._problem.dimN())
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        if self._genHIP:
            filetype = '.cpp'

        print('Load(fftx);', file = script_file)
        print('ImportAll(fftx);', file = script_file) 
        if self._genCuda:
            print('conf := LocalConfig.fftx.confGPU();', file = script_file)
        elif self._genHIP:
            print ( 'conf := FFTXGlobals.defaultHIPConf();', file = script_file )
        else:
            print('conf := LocalConfig.fftx.defaultConf();', file = script_file) 

        print('', file = script_file)  
        print('szcube := [' + ns + ', ' + ns + ', ' + ns + '];', file = script_file)
        print('', file = script_file)    
        print('symvar := var("amplitudes", TPtr(TReal));', file = script_file)
        print('name := "' + nameroot + '";', file = script_file)
        print('domain := MDPRDFT(szcube, -1).dims()[1];', file = script_file)
        print('', file = script_file)
        print('t := TFCall(IMDPRDFT(szcube, 1) * StepPhase_Pointwise(domain, symvar, V(fdiv(1.0, Product(szcube)))) * MDPRDFT(szcube, -1),', file = script_file)
        print('    rec(fname := name, params := [symvar]));', file = script_file)
        print('', file = script_file)
        print('opts := conf.getOpts(t);', file = script_file)
        if self._genCuda or self._genHIP:
            print ( 'opts.wrapCFuncs := true;', file = script_file )
        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)
        print('Add(opts.includes, "<float.h>");',  file = script_file)
        print('tt := opts.tagIt(t);', file = script_file)
        print('', file = script_file)
        print('c := opts.fftxGen(tt);', file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print('', file = script_file)
        
    
