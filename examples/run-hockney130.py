from snowwhite.hockneysolver import *
import numpy as np

p1 = HockneyProblem(130,33,96)
s1 = HockneySolver(p1, {SW_OPT_PLATFORM : SW_CUDA, SW_OPT_KEEPTEMP : False, SW_OPT_PRINTRULETREE : True})

input_data = s1.buildTestInput()

output_Py = s1.runDef(input_data)
output_C = s1.scale(s1.solve(input_data))

diff = np.max ( np.absolute (  output_Py - output_C ))
print ( 'Max Diff between Python/C = ' + str(diff) )