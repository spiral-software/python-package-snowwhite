
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
        
        typ = 'd'
        self._ftype = np.double
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'f'
            self._ftype = np.single
        
        n = str(problem.dimN())
        c = "_";
        namebase = typ + "mdrconv" + c + n
            
        opts[SW_OPT_METADATA] = True
        
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
        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)
        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
    
    def buildTestInput(self, shift = (1,1,1), target=(0,0,0)):
        """ Build test input cube """
        
        xp = cp if self._genCuda else np
        n = self._problem.dimN()
        
        start = (n-shift[0]+target[0],n-shift[1]+target[1],n-shift[2]+target[2])
        
        
        testSrc = xp.zeros((n,n,n)).astype(self._ftype)
        testSrc[start] = 1.0
        
        symIn = xp.zeros((n*2,n*2,n*2)).astype(self._ftype)
        symIn[shift] = 1.0
        testSym = xp.fft.rfftn(symIn)
        
        return (testSrc, testSym)
    
    def _setFunctionMetadata(self, obj):
        obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_MDRCONV
     

    