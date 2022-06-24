#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_simple_loops_c(int n, int m, double *scalar, double * restrict v_vector, 
  double * restrict v_tensor) {
  
  int i, j;
  /* Array casts for pointer arguments */
  double (*vector) = (double (*)) v_vector;
  double (*tensor)[n] = (double (*)[n]) v_tensor;
  
  // For testing, the operation is:
  for (i = 1; i <= n; i += 1) {
    vector[i - 1] = vector[i - 1] + tensor[1 - 1][i - 1] + 1.0;
  }
  
  for (j = 1; j <= m; j += 1) {
    for (i = 1; i <= n; i += 1) {
      tensor[j - 1][i - 1] = 10.*j + i;
    }
  }
  return 0;
}
