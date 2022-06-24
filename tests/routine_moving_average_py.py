import dace
import numpy as np
length = dace.symbol("length")
@dace.program
def routine_moving_average_py(data_in: dace.float64[length], data_out: dace.float64[length]):

  
  data_out[1 - 1] = (data_in[1 - 1] + data_in[2 - 1]) / 2.0
  
  for i in dace.map[2:length - 1+1]:
    # TODO: range check prohibits this for some reason
    incr = 1.0
    divisor = 2.0
    if i > 1:
      prev = data_in[i - 1 - 1]
      # divisor = 2.0
    else:
      divisor = divisor - incr
      prev = 0
      # divisor = 1.0
    if i < length:
      next = data_in[i + 1 - 1]
      divisor = divisor + incr
    else:
      next = 0
    data_out[i - 1] = (prev + data_in[i - 1] + next) / divisor
  
  data_out[length - 1] = (data_in[length - 1 - 1] + data_in[length - 1]) / 2.0
