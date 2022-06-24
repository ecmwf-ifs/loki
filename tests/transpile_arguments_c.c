#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_arguments_c(int n, double * restrict v_array, 
  double * restrict v_array_io, int *a, float *b, double *c, int *a_io, float *b_io, 
  double *c_io) {
  
  
  
  int i;
  /* Array casts for pointer arguments */
  double (*array) = (double (*)) v_array;
  double (*array_io) = (double (*)) v_array_io;
  
  for (i = 1; i <= n; i += 1) {
    array[i - 1] = 3.;
    array_io[i - 1] = array_io[i - 1] + 3.;
  }
  
  *a = pow(2, 3);
  *b = (float) 3.2;
  *c = (double) 4.1;
  
  *a_io = *a_io + 2;
  *b_io = *b_io + (float) (3.2);
  *c_io = *c_io + 4.1;
  return 0;
}
