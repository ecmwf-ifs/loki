import dace
import numpy as np
@dace.program
def routine_fixed_loop_py(scalar: dace.float64[1], vector: dace.float64[6], 
  vector_out: dace.float64[6], tensor: dace.float64[4, 6], tensor_out: dace.float64[6, 4]):

  
  # For testing, the operation is:
  for j in dace.map[1:6+1]:
    vector_out[j - 1] = vector[j - 1] + tensor[1 - 1, j - 1] + 1.0
    for i in dace.map[1:4+1]:
      tensor_out[j - 1, i - 1] = tensor[i - 1, j - 1]
