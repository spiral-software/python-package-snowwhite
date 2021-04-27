
from snowwhite import *
import numpy as np
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
        namebase = "batch3ddft" + n + 'x' + b
        
        super(BatchMddftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        n = self._problem.dimN()
        b = self._problem.szBatch()
        
        out = np.empty(shape=(b, n, n, n)).astype(complex)
        
        for i in range(b):
            dft = np.fft.fftn(src[i,:,:,:]) 
            out[i,:,:,:] = dft 
        
        return out
    
       
    def buildTestInput(self):
        n = self._problem.dimN()
        b = self._problem.szBatch()
               
        ret_Py = np.random.rand(b, n, n, n).astype(complex)
        ret_C = ret_Py.view(dtype=np.double)
        
        return ret_Py, ret_C
        
        
    def _trace(self):
        pass

    def _func(self, dst, src):
        """Call the SPIRAL generated main function -- {_namebase}."""
        funcname = self._namebase
        gf = getattr(self._SharedLibAccess, funcname, None)
        if gf != None:
            return gf( dst.ctypes.data_as(ctypes.c_void_p),
                       src.ctypes.data_as(ctypes.c_void_p) )
        else:
            msg = 'could not find function: ' + funcname
            raise RuntimeError(msg)
    
    def solve(self, src):
        """Call SPIRAL-generated function."""

        n = self._problem.dimN()
        b = self._problem.szBatch()
        dst = np.zeros((b*n**3 * 2), dtype=np.double)
        self._func(dst, src)
        return dst

    def _writeScript(self, script_file):
        nameroot = self._namebase
        filename = nameroot
        filetype = '.c'
        if self._genCuda:
            nameroot = nameroot + '_cu'
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
        if self._printRuleTree:
            print("opts.printRuleTree := true;", file = script_file)
        print('', file = script_file)  

        print('tt := opts.tagIt(t);', file = script_file)
        print('c := opts.fftxGen(tt);', file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print('', file = script_file)
        
    def _writeCudaHost(self):
        """ Write CUDA host code """
        
        # Python interface to C libraries does not handle mangled names from CUDA/C++ compiler
        
        n = self._problem.dimN()
        b = self._problem.szBatch()
        r = 3 # rank set to 3 for now
        
        inSzStr  = str(2 * (n**r) * b)
        outSzStr = str(2 * (n**r) * b)
        
        cu_hostFileName = self._namebase + '_host.cu'
        cu_hostFile = open(cu_hostFileName, 'w')
        
        genby = 'Host-to-Device C/CUDA Wrapper generated by ' + self.__class__.__name__
        print('/*', file=cu_hostFile)
        print(' * ' + genby, file=cu_hostFile)
        print(' */', file=cu_hostFile)
        print('', file=cu_hostFile)
        
        print('#include <helper_cuda.h> \n', file=cu_hostFile)
        
        print('extern void init_' + self._namebase + '_cu();', file=cu_hostFile)
        
        print('extern void ' + self._namebase + '_cu' + '(double  *Y, double  *X);', file=cu_hostFile)
        print('extern void destroy_' + self._namebase + '_cu();\n', file=cu_hostFile)
        print('double  *dev_in, *dev_out; \n', file=cu_hostFile)
        print('extern "C" { \n', file=cu_hostFile)
        print('void init_' + self._namebase + '()' + '{', file=cu_hostFile)
        
        print('    cudaMalloc( &dev_in,  sizeof(double) * ' + inSzStr + ');', file=cu_hostFile)
        print('    cudaMalloc( &dev_out, sizeof(double) * ' + outSzStr +'); \n', file=cu_hostFile)
        print('    init_' + self._namebase + '_cu();', file=cu_hostFile)
        print('} \n', file=cu_hostFile)
        
        print('void ' + self._namebase + '(double  *Y, double  *X) {', file=cu_hostFile)
        print('    cudaMemcpy ( dev_in, X, sizeof(double) * ' + inSzStr + ', cudaMemcpyHostToDevice);', file=cu_hostFile)
        print('    ' + self._namebase + '_cu(dev_out, dev_in);', file=cu_hostFile)
        print('    checkCudaErrors(cudaGetLastError());', file=cu_hostFile)
        print('    cudaMemcpy ( Y, dev_out, sizeof(double) * ' + outSzStr + ', cudaMemcpyDeviceToHost);', file=cu_hostFile)
        print('} \n', file=cu_hostFile)
        
        print('void destroy_' + self._namebase + '() {', file=cu_hostFile)
        print('    cudaFree(dev_out);', file=cu_hostFile)
        print('    cudaFree(dev_in); \n', file=cu_hostFile)
        print('    destroy_' + self._namebase + '_cu();', file=cu_hostFile)
        print('} \n', file=cu_hostFile)
        print('}', file=cu_hostFile)
        
        cu_hostFile.close()



