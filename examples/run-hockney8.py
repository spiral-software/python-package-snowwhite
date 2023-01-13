#! python

"""
usage: run-hockney8.py

  (no arguments)
  
size 8 Hockney convolution on CPU
"""

from snowwhite.hockneysolver import *
import numpy as np

p1 = HockneyProblem(8,3,5)
s1 = HockneySolver(p1, {SW_OPT_PLATFORM : SW_CPU, SW_OPT_PRINTRULETREE : True})

input_data = s1.buildTestInput()

output_Py = s1.runDef(input_data)
output_C = s1.scale(s1.solve(input_data))

diff = np.max ( np.absolute (  output_Py - output_C ))
print ( 'Max Diff between Python/C = ' + str(diff) )