
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np
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
        
        n = str(problem.dimN())
        c = '_'
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = 'dft_fwd' + c + n
        else:
            namebase = 'dft_inv' + c + n
            
        opts[SW_OPT_METADATA] = True
            
        super(DftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        ##  Python performs an FFT in the inverse direction to SPIRAL.  Thus, to do
        ##  a direct comparison we must scale the Python IFFT by the vector length
        ##  before comparing with the SPIRAL FFT
        N = self._problem.dimN()
        if self._problem.direction() == SW_FORWARD:
            FFT = np.fft.fft ( src )
        else:
            FFT = np.fft.ifft ( src )                    ##  * N  ## (no scaling?)

        ##  PW = FFT * symbol
        ##  IFFT = np.fft.irfftn(PW)
        ##  D = IFFT[N-Nd:N, N-Nd:N, N-Nd:N]
        return FFT
        
    def _trace(self):
        pass

    def solve(self, src):
        """Call SPIRAL-generated function."""
        ##  print('DftSolver.solve:')
        n = self._problem.dimN()
        dst = np.zeros(n).astype(complex)
        self._func(dst, src)
        return dst

    def _writeScript(self, script_file):
        nameroot = self._namebase
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        
        
        print("opts := SpiralDefaults;", file = script_file)
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




