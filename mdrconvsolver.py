
from snowwhite import *
import numpy as np
import ctypes
import sys

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

class MdrconvProblem(SWProblem):
    """Define Mdrconv problem."""

    def __init__(self, n):
        """Setup problem specifics for Mdrconv solver.
        
        Arguments:
        n      -- dimension of input/output cube
        """
        super(MdrconvProblem, self).__init__()
        self._n = n

    def dimN(self):
        return self._n


class MdrconvSolver(SWSolver):
    def __init__(self, problem: MdrconvProblem, opts = {}):
        if not isinstance(problem, MdrconvProblem):
            raise TypeError("problem must be an MdrconvProblem")
        gpu = opts.get(SW_OPT_CUDA, False)
        self._symbol = self._buildSymbol(problem, gpu)
        n = str(problem.dimN())
        c = "_";
        namebase = "Mdrconv" + c + n
        super(MdrconvSolver, self).__init__(problem, namebase, opts)

    def _buildSymbol(self, problem, gpu):
        """ Build symbol (build is in order x-->y-->z) """
        
        xp = cp if gpu else np
        n = problem.dimN()
        dims = (2*n, 2*n, n + 1)
        fbox = xp.ones(dims, np.float64)
        return fbox.astype(np.complex128)
            
    def runDef(self, src):
        """Solve using internal Python definition."""

        # Mdrconv problem dimensions
        N = self._problem.dimN() * 2
        Ns = self._problem.dimN()
        Nd = self._problem.dimN()
        
        # Mdrconv operations
        In = self.embedCube(N, src, Ns) # zero pad input data 
        FFT = self.rfftn(In)            # execute real forward dft on rank 3 data      
        P = self.pointwise(FFT, self._symbol) # execute pointwise operation
        IFFT = self.irfftn(P, shape=In.shape)  # execute real backward dft on rank 3 data
        D = self.extract(IFFT, N, Nd)   # extract data from corner cube
        return D
    
    def solve(self, src):
        """Call SPIRAL-generated code"""
        
        xp = xp = cp if self._genCuda else np

        N = self._problem.dimN()
        dst = xp.zeros((N,N,N), src.dtype)
        self._func(dst, src, self._symbol)
        return dst

    def scale(self, d):
        N = 2 * self._problem.dimN()
        return (d / N**3)
 
    def _func(self, dst, src, sym):
        """Call the SPIRAL generated main function"""
        return self._MainFunc( 
                    dst.ctypes.data_as(ctypes.c_void_p),
                    src.ctypes.data_as(ctypes.c_void_p),
                    sym.ctypes.data_as(ctypes.c_void_p) )


    def _writeScript(self, script_file):
        nameroot = self._namebase
        filename = nameroot
        filetype = '.cu' if self._genCuda else '.c'            
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file)
        print("", file = script_file)
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file)
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file)
        print("", file = script_file)
        print('t := let(symvar := var("sym", TPtr(TReal)),', file = script_file)
        print("    TFCall(", file = script_file)
        print("        Compose([", file = script_file)
        for i in range(len(self._callGraph)):
            print("            " + self._callGraph[i], file = script_file)
        print("        ]),", file = script_file)
        print('        rec(fname := "' + nameroot + '", params := [symvar])', file = script_file)
        print("    )", file = script_file)
        print(");", file = script_file)
        print("", file = script_file)
        print("opts := conf.getOpts(t);", file = script_file)
        if self._genCuda:
            print('opts.wrapCFuncs := true;', file = script_file)
        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + ".icode" + '", c);', file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
    
    def buildTestInput(self):
        """ Build test input cube """
        
        xp = cp if self._genCuda else np
        n = self._problem.dimN()
        return xp.random.rand(n,n,n).astype(np.float64)
     

    