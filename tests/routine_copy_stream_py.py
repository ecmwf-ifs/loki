import dace
import numpy as np
length = dace.symbol("length")
@dace.program
def routine_copy_stream_py(alpha: dace.int32[1], vector_in: dace.int32[length], 
  vector_out: dace.int32[length]):

  
  for i in dace.map[1:length+1]:
    vector_out[i - 1] = vector_in[i - 1] + alpha[1 - 1]
