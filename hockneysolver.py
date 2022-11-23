
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np
import ctypes
import sys

class HockneyProblem(SWProblem):
    """Define Hockney problem."""

    def __init__(self, n, ns, nd):
        """Setup problem specifics for Hockney solver.
        
        Arguments:
        n      -- dimension of full cube
        ns     -- dimension of input cube
        nd     -- dimension of output cube
        """
        super(HockneyProblem, self).__init__([n,n,n])
        self._ns = ns
        self._nd = nd
        
    def dimNS(self):
        return self._ns
        
    def dimND(self):
        return self._nd


class HockneySolver(SWSolver):
    def __init__(self, problem: HockneyProblem, opts = {}):
        if not isinstance(problem, HockneyProblem):
            raise TypeError("problem must be a HockneyProblem")
        n = str(problem.dimN())
        ns = str(problem.dimNS())
        nd = str(problem.dimND())
        
        self._symbol = self._buildSymbol(problem)

        c = "_";
        namebase = "hockney" + c + n + c + ns + c + nd
        super(HockneySolver, self).__init__(problem, namebase, opts)

    def _buildSymbol(self, problem):
        """ Build symbol (build is in order x-->y-->z) """
        
        n = problem.dimN()
        nf = n//2+1
        pi = np.pi
        np.random.seed(0)
        # generate initial symbol (one octant)
        sym_oct = np.array([[[(1/(4*pi*((n/2-i)*(n/2-i)+(n/2-j)*(n/2-j)+(n/2-k)*(n/2-k)))) \
                if (i<n/2 or j<n/2 or k<n/2) else 0 \
                for i in range(nf)]for j in range(nf)] \
                for k in range(nf)]).astype(complex)
         
        # reflection of S: tmp1[x,y,z] = S[(Nf-1)-x, y, z]
        tmp1 = np.flip(sym_oct, axis=0) # 1st reflection
        tmp2 = np.concatenate([sym_oct,tmp1], axis=0) # 1st stitch
        if n%2==0: # even N case
            tmp3 = np.delete(tmp2, [nf, -1], axis=0) # drop 2 planes
        else: # odd N case
            tmp3 = np.delete(tmp2, [nf], axis=0) # drop 1 plane
        
        # reflection of tmp1: tmp2[x,y,z] = tmp1[x, (Nf-1)-y, z]
        tmp4 = np.flip(tmp3, axis=1) # 2nd reflection
        tmp5 = np.concatenate([tmp3,tmp4], axis=1) # 2nd stitch
        if n%2==0: # even N case
            return np.delete(tmp5, [nf,-1], axis=1) # drop 2 planes
        else: # odd N case
            return np.delete(tmp5, [nf], axis=1) # drop 1 plane
            
    def runDef(self, src):
        """Solve using internal Python definition."""

        # Hockney problem dimensions
        N = self._problem.dimN()
        Ns = self._problem.dimNS()
        Nd = self._problem.dimND()
        
        # Hockney operations
        In = self.zeroEmbedBox(src, ((0,N-Ns),)) # zero pad input data 
        FFT = self.rfftn(In)            # execute real forward dft on rank 3 data 
        P = self.pointwise(FFT, self._symbol) # execute pointwise operation
        IFFT = self.irfftn(P, shape=In.shape)  # execute real backward dft on rank 3 data
        D = self.extract(IFFT, N, Nd)   # extract data from corner cube
        return D
    
    def solve(self, src, dst=None):
        """Call SPIRAL-generated code"""
        
        if type(dst) == type(None):
            Nd = self._problem.dimND()
            dst = np.zeros((Nd,Nd,Nd), dtype=np.double)

        # swapaxes was necessary b/c C interprets symbol in y-->x-->z order
        self._func(dst, src, np.swapaxes(self._symbol, axis1=0, axis2=1))
        
        return dst

    def scale(self, d):
        N = self._problem.dimN()
        return (d / N**3)
 
    def _func(self, dst, src, sym):
        """Call the SPIRAL generated main function"""
        return self._MainFunc( 
                    dst.ctypes.data_as(ctypes.c_void_p),
                    src.ctypes.data_as(ctypes.c_void_p),
                    sym.ctypes.data_as(ctypes.c_void_p) )


    def _writeScript(self, script_file):
        n = self._problem.dimN()
        ns = self._problem.dimNS()
        nd = self._problem.dimND()
        nameroot = self._namebase
        filename = nameroot
        nnn = '[' + str(n) + ',' + str(n) + ',' + str(n) + ']'
        ndrange = '[' + str(n-nd) + '..' + str(n-1) + ']'
        ndr3D = '[' + ndrange + ',' + ndrange + ',' + ndrange + ']'
        nnhlf3D = str(2*n*n*(n//2+1))
        nsrange = '[0..' + str(ns-1) + ']'
        nsr3D = '['+nsrange+','+nsrange+','+nsrange+']'
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
            
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file) 
        print("Import(realdft);", file = script_file)
        print("", file = script_file)
        if self._genCuda:
            print("conf := FFTXGlobals.confHockneyMlcCUDADevice();", file = script_file)
            print("opts := FFTXGlobals.getOpts(conf);", file = script_file)
            print("opts.devFunc := true;", file = script_file)
            print('opts.wrapCFuncs := true;', file = script_file)
        else:
            print("conf := FFTXGlobals.mdRConv();", file = script_file)
            print("opts := FFTXGlobals.getOpts(conf);", file = script_file)
            print("opts.preProcess := (self, t) >> t;", file = script_file)
        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)
        print("", file = script_file)
        print('t := let(symvar := var("sym", TPtr(TReal)),', file = script_file)
        print("    TFCall(", file = script_file)
        print("        Compose([", file = script_file)
        for i in range(len(self._callGraph)):
            print("            " + self._callGraph[i], file = script_file)
        print("        ]),", file = script_file)
        print('        rec(fname := "' + nameroot + '", params := [symvar])', file = script_file)
        print("    ).withTags(opts.tags)", file = script_file)
        print(");", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(t);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)

    
    def buildTestInput(self):
        """ Build initial input using synthetic values of size (Ns, Ns, Ns) with NO zero padding """

        ns = self._problem.dimNS()
		
        ret = np.array([[[(i*ns**2+1)+(j*ns)+(k) for i in range(ns)]for j in range(ns)] \
			for k in range(ns)]).astype(np.float)

        return ret
     
     
    