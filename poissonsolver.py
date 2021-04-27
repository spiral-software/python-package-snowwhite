
from swsolver import *
import numpy as np
import ctypes
import sys

class PoissonProblem(swsolver.SWProblem):
    """Define Poisson problem."""

    def __init__(self, n):
        """Setup problem specifics for Poisson solver.
        
        Arguments:
        n      -- dimension size for (n,n,n) cube

        """
        super(PoissonProblem, self).__init__()
        self._n = n
    
    def dimN(self):
        return self._n


class PoissonSolver(swsolver.SWSolver):
    def __init__(self, problem: PoissonProblem):
        if not isinstance(problem, PoissonProblem):
            raise TypeError("problem must be a PoissonProblem")
        super(PoissonSolver, self).__init__(problem)
        pass

    def runDef(self, src, sym):
        """Solve using internal Python definition."""
        # Poisson operations 
        fft = self.fftn(src) # complex forward dft on rank 3 data
        p = self.pointwise(fft, sym) # execute pointwise operation
        ifft = self.ifftn(p) #  complex inverse dft on rank 3 data
        return ifft
    
    def fftn(self, x):
        """ Forward complex multi-dimensional DFT using NumPy """
        ret = np.fft.fftn(x)
        return ret
    
    def pointwise(self, x, y):
        """ Pointwise array multiplication """
        ret = x * y
        return ret
    
    def ifftn(self, x):
        """ Inverse complex multi-dimensional DFT using NumPy """
        ret = np.fft.ifftn(x) # executes x, then y, then z
        return ret
    
    def buildTestInput(self):
        n = self._problem.dimN()
        # generic input data
        input_data = np.empty(shape=(n, n, n), dtype=np.complex)
        cntr = 0
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    input_data[i,j,k] = cntr+1 + ((cntr +1) *1j)
                    cntr += 1
        return input_data
            