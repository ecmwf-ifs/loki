import dace
import numpy as np
n = dace.symbol("n")
@dace.program
def routine_copy_py(x: dace.float64[n], y: dace.float64[n]):

  
  for i in range(1, n + 1):
    y[i - 1] = x[i - 1]
