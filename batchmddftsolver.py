
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

class BatchMddftProblem(SWProblem):
    """Define Batch MDDFT problem."""

    def __init__(self, n, batchSz):
        """Setup problem specifics for Batch MDDFT solver.
        
        Arguments:
        n         -- dimension of 3D DFT Cube
        batchSz   -- batch size
        """
        super(BatchMddftProblem, self).__init__()
        self._n = n
        self._batchSz = batchSz
        
    def dimN(self):
        return self._n
        
    def szBatch(self):
        return self._batchSz
        

class BatchMddftSolver(SWSolver):
    def __init__(self, problem: BatchMddftProblem, opts = {}):
        if not isinstance(problem, BatchMddftProblem):
            raise TypeError("problem must be a BatchMddftProblem")
 
        n = str(problem.dimN())
        b = str(problem.szBatch())
        typ = 'z'
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
        namebase = typ + 'batch3ddft' + n + 'x' + b
        
        super(BatchMddftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        n = self._problem.dimN()
        b = self._problem.szBatch()
        
        xp = get_array_module(src)
        
        out = xp.empty(shape=(b, n, n, n)).astype(complex)
        
        for i in range(b):
            dft = xp.fft.fftn(src[i,:,:,:]) 
            out[i,:,:,:] = dft 
        
        return out
    
       
    def buildTestInput(self):
        n = self._problem.dimN()
        b = self._problem.szBatch()
        
        xp = cp if self._genCuda else np
               
        ret_Py = xp.random.rand(b, n, n, n).astype(complex)
        ret_C = ret_Py.view(dtype=np.double)
        
        return ret_Py, ret_C
        
        
    def _trace(self):
        pass
    
    def solve(self, src):
        """Call SPIRAL-generated function."""
        
        xp = get_array_module(src)

        n = self._problem.dimN()
        b = self._problem.szBatch()
        dst = xp.zeros((b*n**3 * 2), dtype=np.double)
        self._func(dst, src)
        return dst

    def _writeScript(self, script_file):
        nameroot = self._namebase
        filename = nameroot
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
            
        n = str(self._problem.dimN())
        b = str(self._problem.szBatch())
        nnn = '[' + n + ',' + n + ',' + n + ']'
        
        print('Load(fftx);', file = script_file)
        print('ImportAll(fftx);', file = script_file) 
        print('', file = script_file)
            
        print('t := let(batch := ' + b + ',', file = script_file)
        print('    apat := When(true, APar, AVec),', file = script_file)
        print('    ns := ' + nnn + ',', file = script_file)
        print('    k := -1,', file = script_file)
        print('    name := "' + nameroot + '",', file = script_file)
        print('    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)),', file = script_file)
        print('        rec(fname := name, params := []))', file = script_file)
        print(');', file = script_file)
        print('', file = script_file)
        
        if self._genCuda:
            print('conf := LocalConfig.fftx.confGPU();', file = script_file)
        else:
            print('conf := LocalConfig.fftx.defaultConf();', file = script_file)
        print('opts := conf.getOpts(t);', file = script_file)
        if self._genCuda:
            print('opts.wrapCFuncs := true;', file = script_file)
        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)
        print('', file = script_file)  

        print('tt := opts.tagIt(t);', file = script_file)
        print('c := opts.fftxGen(tt);', file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print('', file = script_file)
        
    