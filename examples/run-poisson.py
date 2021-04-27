from snowwhite.poissonsolver import *
import numpy as np

n=256 # problem size (n,n,n)
p1 = PoissonProblem(n)
s1 = PoissonSolver(p1)

# generic input data
input_data = s1.buildTestInput()
# generic symbol data
symbol = np.ones(shape=(n,n,n), dtype=np.complex128) # symbol data
# poisson solver
output_Py = s1.runDef(input_data, symbol)

print('Inf Norm, input vs output: ', np.max(np.abs(input_data - output_Py)))
