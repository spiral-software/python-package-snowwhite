
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


class MddftProblem(SWProblem):
    """Define Multi-dimention DFT problem."""

    def __init__(self, ns, k=SW_FORWARD):
        """Setup problem specifics for MDDFT solver.
        
        Arguments:
        ns     -- dimensions of MDDFT
        k      -- direction
        """
        super(MddftProblem, self).__init__(ns, k)
        

class MddftSolver(SWSolver):
    def __init__(self, problem: MddftProblem, opts = {}):
        if not isinstance(problem, MddftProblem):
            raise TypeError("problem must be an MddftProblem")
        
        typ = 'z'
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
        ns = 'x'.join([str(n) for n in problem.dimensions()])
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = typ + 'mddft_fwd_' + ns
        else:
            namebase = typ + 'mddft_inv_' + ns
        
        if opts.get(SW_OPT_COLMAJOR, False):
            namebase = namebase + '_F'
            
        opts[SW_OPT_METADATA] = True
                    
        super(MddftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        xp = get_array_module(src)

        if self._problem.direction() == SW_FORWARD:
            FFT = xp.fft.fftn ( src )
        else:
            FFT = xp.fft.ifftn ( src ) 

        return FFT
        
    def _trace(self):
        pass

    def solve(self, src, dst=None):
        """Call SPIRAL-generated function."""
   
        if type(dst) == type(None):
            xp = get_array_module(src)
            nt = tuple(self._problem.dimensions())
            ordc = 'F' if self._colMajor else 'C'
            dst = xp.zeros(nt, src.dtype,  order=ordc)
            
        self._func(dst, src)
        if self._problem.direction() == SW_INVERSE:
            xp = get_array_module(dst)
            xp.divide(dst, xp.size(dst), out=dst)
        return dst

    def _writeScript(self, script_file):
        filename = self._namebase
        nameroot = self._namebase
        dims = str(self._problem.dimensions())
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        if self._genHIP:
            filetype = '.cpp'
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file) 
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file) 
        elif self._genHIP:
            print ( 'conf := FFTXGlobals.defaultHIPConf();', file = script_file )
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file) 

        print('', file = script_file)
        print("t := let(ns := " + dims + ",", file = script_file) 
        print('    name := "' + nameroot + '",', file = script_file)
        # -1 is inverse for Numpy and forward (1) for Spiral
        if self._colMajor:
            print("    TFCall(TRC(TColMajor(MDDFT(ns, " + str(self._problem.direction()) + "))), rec(fname := name, params := []))", file = script_file)
        else:
            print("    TFCall(TRC(MDDFT(ns, " + str(self._problem.direction()) + ")), rec(fname := name, params := []))", file = script_file)
        print(");", file = script_file)        

        print('', file = script_file)
        print("opts := conf.getOpts(t);", file = script_file)
        if self._genCuda or self._genHIP:
            print('opts.wrapCFuncs := true;', file = script_file)

        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)

        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)

        print('Add(opts.includes, "<float.h>");',  file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
        
    def _setFunctionMetadata(self, obj):
        obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_MDDFT
        
