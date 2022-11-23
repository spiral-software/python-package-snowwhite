"""
MDPRDFT Module
"""


from snowwhite import *
from snowwhite.swsolver import *
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


class MdprdftProblem(SWProblem):
    """
    Multi-dimention Packed Real DFT (MDPRDFT) problem.
            
    Constructor: MdprdftProblem(ns, k=SW_FORWARD)
        ns -- shape (tuple) of MDPRDFT box of reals (input for forward, output for inverse)
        k  -- direction, SW_FORWARD or SW_INVERSE
    """
 
    def __init__(self, ns, k=SW_FORWARD):
        """Setup problem specifics for MDPRDFT solver."""
        
        super(MdprdftProblem, self).__init__(ns, k)
        # last dimension of complex component is smaller
        cxns = ns.copy()
        z = cxns.pop()
        cxns.append(z // 2 + 1)
        self._cxns = cxns

    def dimensionsCX(self):
        return self._cxns
        

class MdprdftSolver(SWSolver):
    def __init__(self, problem: MdprdftProblem, opts = {}):
        if not isinstance(problem, MdprdftProblem):
            raise TypeError("problem must be an MddftProblem")
        
        typ = 'z'
        self._ftype = np.double
        self._cxtype = np.cdouble
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
            self._ftype = np.single
            self._cxtype = np.csingle
        ns = 'x'.join([str(n) for n in problem.dimensions()])
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = typ + 'mdprdft_' + ns
        else:
            namebase = typ + 'imdprdft_' + ns
        
        if opts.get(SW_OPT_COLMAJOR, False):
            namebase = namebase + '_F'
            
        opts[SW_OPT_METADATA] = True
                    
        super(MdprdftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        xp = get_array_module(src)

        if self._problem.direction() == SW_FORWARD:
            dst = xp.fft.rfftn ( src )
        else:
            dst = xp.fft.irfftn ( src, tuple(self._problem.dimensions()) )

        return dst
        
    def _trace(self):
        pass

    def solve(self, src, dst=None):
        """Call SPIRAL-generated function."""
        
        if type(dst) == type(None):
            xp = get_array_module(src)
            if self._problem.direction() == SW_FORWARD:
                nt = tuple(self._problem.dimensionsCX())
                rtype = self._cxtype
            else:
                nt = tuple(self._problem.dimensions())
                rtype = self._ftype
            ordc = 'F' if self._colMajor else 'C'
            dst = xp.zeros(nt, rtype,  order=ordc)
            
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
            
        xform = "MDPRDFT"
        if self._problem.direction() == SW_INVERSE:
            xform = "IMDPRDFT"
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file) 
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file) 
        elif self._genHIP:
            print ( 'conf := FFTXGlobals.defaultHIPConf();', file = script_file )
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file) 

        print("t := let(ns := " + dims + ",", file = script_file) 
        print('    name := "' + nameroot + '",', file = script_file)
        # -1 is inverse for Numpy and forward (1) for Spiral
        if self._colMajor:
            print("    TFCall(TColMajor(" + xform + "(ns, " + str(self._problem.direction()) + ")), rec(fname := name, params := []))", file = script_file)
        else:
            print("    TFCall(" + xform + "(ns, " + str(self._problem.direction()) + "), rec(fname := name, params := []))", file = script_file)
        print(");", file = script_file)        

        print("opts := conf.getOpts(t);", file = script_file)
        if self._genCuda or self._genHIP:
            print('opts.wrapCFuncs := true;', file = script_file)

        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)

        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
    
    def _setFunctionMetadata(self, obj):
        obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_MDPRDFT
     
        
    
