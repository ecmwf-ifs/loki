#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_expressions_c(int n, double scalar, double * restrict v_vector) {
  
  int i;
  /* Array casts for pointer arguments */
  double (*vector) = (double (*)) v_vector;
  
  vector[1 - 1] = scalar;
  for (i = 2; i <= n; i += 1) {
    vector[i - 1] = vector[i - 1 - 1] - (-scalar);
  }
  return 0;
}
