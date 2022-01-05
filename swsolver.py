
import snowwhite as sw

import datetime
import subprocess
import os
import sys

import tempfile
import shutil

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import ctypes
import sys

SW_OPT_CUDA             = 'cuda'
SW_OPT_KEEPTEMP         = 'keeptemp'
SW_OPT_MPI              = 'mpi'
SW_OPT_PRINTRULETREE    = 'printruletree'
SW_OPT_REALCTYPE        = 'realctype'

SW_FORWARD  = 1
SW_INVERSE  = -1

class SWProblem:
    """Base class for SnowWhite problem."""
    
    def __init__(self):
        pass
        

class SWSolver:
    """Base class for SnowWhite solver."""
    
    def __init__(self, problem: SWProblem, namebase = 'func', opts = {}):
        self._problem = problem
        self._opts = opts
        self._genCuda = self._opts.get(SW_OPT_CUDA, False)
        self._keeptemp = self._opts.get(SW_OPT_KEEPTEMP, False)
        self._withMPI = self._opts.get(SW_OPT_MPI, False)
        self._printRuleTree = self._opts.get(SW_OPT_PRINTRULETREE, False)
        self._runResult = None
        self._tracingOn = False
        self._callGraph = []
        self._SharedLibAccess = None
        self._MainFunc = None
        self._spiralname = 'spiral'
        
        # find and possibly create the subdirectory of temp dirs
        moduleDir = os.path.dirname(os.path.realpath(__file__))
        self._libsDir = os.path.join(moduleDir, '.libs')
        os.makedirs(self._libsDir, mode=0o777, exist_ok=True)
        
        if self._genCuda:
            self._namebase = namebase + '_cu'
        else:
            self._namebase = namebase
        if sys.platform == 'win32':
            libext = '.dll'
        else:
            libext = '.so'
        sharedLibFullPath = os.path.join(self._libsDir, 'lib' + self._namebase + libext)         

        if not os.path.exists(sharedLibFullPath):
            self._setupCFuncs(self._namebase)

        self._SharedLibAccess = ctypes.CDLL (sharedLibFullPath)
        self._MainFunc = getattr(self._SharedLibAccess, self._namebase)
        if self._MainFunc == None:
            msg = 'could not find function: ' + funcname
            raise RuntimeError(msg)
        self._initFunc()

    def __del__(self):
        self._destroyFunc()
    
    def solve(self):
        raise NotImplementedError()

    def runDef(self):
        raise NotImplementedError()
        
    def _writeScript(self, script_file):
        raise NotImplementedError()
    
    def _genScript(self, filename : str):
        print("Tracing Python description to generate SPIRAL script");
        self._trace()
        try:
            script_file = open(filename, 'w')
        except:
            print('Error: Could not open ' + filename + ' for writing')
            return
        timestr = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        print(file = script_file)
        print("# SPIRAL script generated by " + type(self).__name__, file = script_file)
        print('# ' + timestr, file = script_file)
        print(file = script_file)
        self._writeScript(script_file)
        script_file.close()
        
    def _callSpiral(self, script):
        """Run SPIRAL with script as input."""
        if self._genCuda:
            print("Generating CUDA code");
        else:
            print("Generating C code");
        if sys.platform == 'win32':
            spiralexe = self._spiralname + '.bat'
            self._runResult = subprocess.run([spiralexe,'<',script], shell=True, capture_output=True)
        else:
            spiralexe = self._spiralname
            cmd = spiralexe + ' < ' + script
            self._runResult = subprocess.run(cmd, shell=True)

    def _callCMake (self, basename):
        ##  create a temporary work directory in which to run cmake
        ##  Assumes:  SPIRAL_HOME is defined (environment variable) or override on command line
        ##  FILEROOT = basename;
        
        print("Compiling and linking C code");
        
        cwd = os.getcwd()
        
        # get module CMakeLists if none exists in current directory
        if not os.path.exists('CMakeLists.txt'):
            module_dir = os.path.dirname(__file__)
            cmfile = os.path.join(module_dir, 'CMakeLists.txt')
            shutil.copy(cmfile, os.getcwd())
            
        tempdir = tempfile.mkdtemp(None, None, cwd)
        os.chdir(tempdir)

        cmake_defroot = '-DFILEROOT:STRING=' + basename
        if sys.platform == 'win32':
            ##  NOTE: Ensure Python installed on Windows is 64 bit
            cmd = ['cmake', cmake_defroot]
            if self._genCuda:
                cmd.append('-DHASCUDA=1')
            if self._withMPI:
                cmd.append('-DHASMPI=1')
            cmd.append('-DPY_LIBS_DIR=' + self._libsDir)
            cmd = cmd + ['..', '&&', 'cmake', '--build', '.', '--config', 'Release', '--target', 'install']
            print(cmd)
            self._runResult = subprocess.run (cmd, shell=True, capture_output=False)
        else:
            cmd = 'cmake ' + cmake_defroot
            if self._genCuda:
                cmd = cmd + ' -DHASCUDA=1'
            if self._withMPI:
                cmd = cmd + ' -DHASMPI=1'
            cmd = cmd + ' -DPY_LIBS_DIR=' + self._libsDir
            cmd = cmd + ' .. && make install'
            self._runResult = subprocess.run(cmd, shell=True)

        os.chdir(cwd)

        if self._runResult.returncode == 0 and not self._keeptemp:
            shutil.rmtree(tempdir, ignore_errors=True)
            
    def _writeCudaHost(self):
        pass
            
    def _setupCFuncs(self, basename):
        script = basename + ".g"
        self._genScript(script)
        self._callSpiral(script)
        if self._genCuda:
            self._writeCudaHost()
        self._callCMake(basename)
        
    def _trace(self):
        """Trace execution for generating Spiral script"""
        self._tracingOn = True
        self._callGraph = []
        src = self.buildTestInput()
        self.runDef(src)
        self._tracingOn = False
        for i in range(len(self._callGraph)-1):
            self._callGraph[i] = self._callGraph[i] + ','

    def _initFunc(self):
        """Call the SPIRAL generated init function"""
        funcname = 'init_' + self._namebase
        gf = getattr(self._SharedLibAccess, funcname, None)
        if gf != None:
            return gf()
        else:
            msg = 'could not find function: ' + funcname
            raise RuntimeError(msg)

    def _func(self, dst, src):
        """Call the SPIRAL generated main function"""
        
        xp = sw.get_array_module(src)
        
        if xp == np: 
            if self._genCuda:
                raise RuntimeError('CUDA function requires CuPy arrays')
            # NumPy array on CPU
            return self._MainFunc( 
                    dst.ctypes.data_as(ctypes.c_void_p),
                    src.ctypes.data_as(ctypes.c_void_p) )
        else:
            if not self._genCuda:
                raise RuntimeError('CPU function requires NumPy arrays')
            # CuPy array on GPU
            srcdev = ctypes.cast(src.data.ptr, ctypes.POINTER(ctypes.c_void_p))
            dstdev = ctypes.cast(dst.data.ptr, ctypes.POINTER(ctypes.c_void_p))
            return self._MainFunc(dstdev, srcdev)

        
    def _destroyFunc(self):
        """Call the SPIRAL generated destroy function"""
        funcname = 'destroy_' + self._namebase
        gf = getattr(self._SharedLibAccess, funcname, None)
        if gf != None:
            return gf()
        else:
            msg = 'could not find function: ' + funcname
            raise RuntimeError(msg)

    def embedCube(self, N, src, Ns):
        retCube = np.zeros(shape=(N, N, N))
        for k in range(Ns):
            for j in range(Ns):
                for i in range(Ns):
                    retCube[i,j,k] = src[i,j,k]
        if self._tracingOn:
            nnn = '[' + str(N) + ',' + str(N) + ',' + str(N) + ']'
            nsrange = '[0..' + str(Ns-1) + ']'
            nsr3D = '['+nsrange+','+nsrange+','+nsrange+']'
            st = 'ZeroEmbedBox(' + nnn + ', ' + nsr3D + ')'
            self._callGraph.insert(0, st)
        return retCube
		        
    def rfftn(self, x):
        """ forward multi-dimensional real DFT """
        ret = np.fft.rfftn(x) # executes z, then y, then x
        if self._tracingOn:
            N = x.shape[0]
            nnn = '[' + str(N) + ',' + str(N) + ',' + str(N) + ']'
            st = 'MDPRDFT(' + nnn + ', -1)'
            self._callGraph.insert(0, st)
        return ret

    def pointwise(self, x, y):
        """ pointwise array multiplication """
        ret = x * y
        if self._tracingOn:
            nElems = np.size(x) * 2
            st = 'RCDiag(FDataOfs(symvar, ' + str(nElems) + ', 0))'
            self._callGraph.insert(0, st)
        return ret

    def irfftn(self, x, shape):
        """ inverse multi-dimensional real DFT """
        ret = np.fft.irfftn(x, s=shape) # executes x, then y, then z
        if self._tracingOn:
            N = x.shape[0]
            nnn = '[' + str(N) + ',' + str(N) + ',' + str(N) + ']'
            st = 'IMDPRDFT(' + nnn + ', 1)'
            self._callGraph.insert(0, st)
        return ret

    def extract(self, x, N, Nd):
        """ Extract output data of dimension (Nd, Nd, Nd) from the corner of cube (N, N ,N) """
        ret = x[N-Nd:N, N-Nd:N, N-Nd:N]
        if self._tracingOn:
            nnn = '[' + str(N) + ',' + str(N) + ',' + str(N) + ']'
            ndrange = '[' + str(N-Nd) + '..' + str(N-1) + ']'
            ndr3D = '[' + ndrange + ',' + ndrange + ',' + ndrange + ']'
            st = 'ExtractBox(' + nnn + ', ' + ndr3D + ')'
            self._callGraph.insert(0, st)
        return ret

