
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

class DftProblem(SWProblem):
    """Define 1D DFT problem."""

    def __init__(self, n, k=SW_FORWARD):
        """Setup problem specifics for 1D DFT solver.
        
        Arguments:
        n      -- dimension of 1D DFT
        k      -- direction
        """
        super(DftProblem, self).__init__([n], k)


class DftSolver(SWSolver):
    def __init__(self, problem: DftProblem, opts = {}):
        if not isinstance(problem, DftProblem):
            raise TypeError("problem must be a DftProblem")
        
        typ = 'z'
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
        n = str(problem.dimN())
        c = '_'
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = typ + 'dft_fwd' + c + n
        else:
            namebase = typ + 'dft_inv' + c + n
            
        opts[SW_OPT_METADATA] = True
            
        super(DftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        xp = get_array_module(src)

        N = self._problem.dimN()
        if self._problem.direction() == SW_FORWARD:
            FFT = xp.fft.fft(src)
        else:
            FFT = xp.fft.ifft(src)

        return FFT
        
    def _trace(self):
        pass

    def solve(self, src, dst=None):
        """Call SPIRAL-generated function."""
        ##  print('DftSolver.solve:')
        if type(dst) == type(None):
            xp = get_array_module(src)
            n = self._problem.dimN()
            dst = xp.zeros(n, src.dtype)
        self._func(dst, src)
        return dst

    def _writeGPUScript(self, script_file):
        filename = self._namebase
        nameroot = self._namebase
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        if self._genHIP:
            filetype = '.cpp'
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file)

        print('', file = script_file)
        
        dft_def = 'DFT(N, ' + str(self._problem.direction()) + ')'
        if self._problem.direction() == SW_INVERSE:
            dft_def = 'Scale(1/N, ' + dft_def + ')'
        
        print('t := let(', file = script_file) 
        print('    name := "' + nameroot + '",', file = script_file)
        print('    N  := ' + str(self._problem.dimN()) + ',', file = script_file)
        print('    TFCall(TRC(TMap(' + dft_def + ', [Ind(1), Ind(1), Ind(1)], AVec, AVec)), rec(fname := name, params := []))', file = script_file)
        print(');', file = script_file)
        
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file) 
        elif self._genHIP:
            print ( 'conf := FFTXGlobals.defaultHIPConf();', file = script_file )
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file) 
        print("opts := conf.getOpts(t);", file = script_file)
        print('opts.wrapCFuncs := true;', file = script_file)

        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)

        print('Add(opts.includes, "<float.h>");',  file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
        
    def _writeScript(self, script_file):
        if self._genCuda or self._genHIP:
            self._writeGPUScript(script_file)
            return
    
        nameroot = self._namebase
        filetype = '.c'
        
        print("opts := SpiralDefaults;", file = script_file)
        
        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)
            
        print("", file = script_file)
        print('n  := ' + str ( self._problem.dimN() )  + ';', file = script_file)
        print("", file = script_file)
        print('nameroot := "' + self._namebase + '";', file = script_file)
        print("", file = script_file)
        if self._problem.direction() == SW_INVERSE:
            print('transform := Scale(1/n, DFT(n, ' + str (self._problem.direction()) + '));', file = script_file)
        else:
            print('transform := DFT(n, ' + str (self._problem.direction()) + ');', file = script_file)
        print('ruletree  := RuleTreeMid(transform, opts);', file = script_file)
        print('code      := CodeRuleTree(ruletree, opts);', file = script_file)
        print('PrintTo("' + nameroot + filetype + '", PrintCode(nameroot, code, opts));', 
            file = script_file)
        print("", file = script_file)    
        
        
    def _setFunctionMetadata(self, obj):
        obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_DFT




