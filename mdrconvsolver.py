
from snowwhite import *
from snowwhite.swsolver import *
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
        super(MdrconvProblem, self).__init__([n,n,n])

    def dimN(self):
        return self.dimensions()[0]


class MdrconvSolver(SWSolver):
    def __init__(self, problem: MdrconvProblem, opts = {}):
        if not isinstance(problem, MdrconvProblem):
            raise TypeError("problem must be an MdrconvProblem")
        n = str(problem.dimN())
        c = "_";
        namebase = "Mdrconv" + c + n
        super(MdrconvSolver, self).__init__(problem, namebase, opts)

        
    def _trace(self):
        """Trace execution for generating Spiral script"""
        self._tracingOn = True
        self._callGraph = []
        (src,sym) = self.buildTestInput()
        self.runDef(src,sym)
        self._tracingOn = False
        for i in range(len(self._callGraph)-1):
            self._callGraph[i] = self._callGraph[i] + ','
            
    def runDef(self, src, sym):
        """Solve using internal Python definition."""

        # Mdrconv problem dimensions
        N = self._problem.dimN() * 2
        Ns = self._problem.dimN()
        Nd = self._problem.dimN()
        
        # Mdrconv operations
        In = self.embedCube(N, src, Ns) # zero pad input data 
        FFT = self.rfftn(In)            # execute real forward dft on rank 3 data      
        P = self.pointwise(FFT, sym) # execute pointwise operation
        IFFT = self.irfftn(P, shape=In.shape)  # execute real backward dft on rank 3 data
        D = self.extract(IFFT, N, Nd)   # extract data from corner cube
        return D
    
    def solve(self, src, sym):
        """Call SPIRAL-generated code"""
        
        xp = xp = cp if self._genCuda else np
                
        N = self._problem.dimN()
        dst = xp.zeros((N,N,N), src.dtype)
        self._func(dst, src, sym)
        return dst

    def scale(self, d):
        N = 2 * self._problem.dimN()
        return (d / N**3)
 
    def _func(self, dst, src, sym):
        """Call the SPIRAL generated main function"""
                
        xp = sw.get_array_module(src)
        
        if xp == np: 
            if self._genCuda or self._genHIP:
                raise RuntimeError('GPU function requires CuPy arrays')
            # NumPy array on CPU
            return self._MainFunc( 
                    dst.ctypes.data_as(ctypes.c_void_p),
                    src.ctypes.data_as(ctypes.c_void_p),
                    sym.ctypes.data_as(ctypes.c_void_p)  )
        else:
            if not self._genCuda and not self._genHIP:
                raise RuntimeError('CPU function requires NumPy arrays')
            # CuPy array on GPU
            dstdev = ctypes.cast(dst.data.ptr, ctypes.POINTER(ctypes.c_void_p))
            srcdev = ctypes.cast(src.data.ptr, ctypes.POINTER(ctypes.c_void_p))
            symdev = ctypes.cast(sym.data.ptr, ctypes.POINTER(ctypes.c_void_p))
            return self._MainFunc(dstdev, srcdev, symdev)
  

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
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
    
    def buildTestInput(self):
        """ Build test input cube """
        
        xp = cp if self._genCuda else np
        n = self._problem.dimN()
        
        testSrc = xp.random.rand(n,n,n).astype(np.float64)
        
        dims = (2*n, 2*n, n+1)
        testSym = xp.zeros(dims, np.complex128)
        z = dims[0]
        y = dims[1]
        x = dims[2]
        for k in range(x):
            for j in range(y):
                for i in range(z):
                    # use NumPy rand() because CuPy version returns array
                    b1 = np.random.random()
                    b2 = np.random.random()
                    v = b1 + (b2 * 1j)
                    testSym[i,j,k] = v
        
        return (testSrc, testSym)
     

    