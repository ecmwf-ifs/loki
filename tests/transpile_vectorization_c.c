#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_vectorization_c(int n, int m, double *scalar, double * restrict v_v1, 
  double * restrict v_v2) {
  
  
  int i;
  int i_matrix_0;
  int i_v1_0;
  int i_matrix_1;
  int i_v2_0;
  double matrix[m][n];
  /* Array casts for pointer arguments */
  double (*v1) = (double (*)) v_v1;
  double (*v2) = (double (*)) v_v2;
  
  for (i_v1_0 = 1; i_v1_0 <= n; i_v1_0 += 1) {
    v1[i_v1_0 - 1] = *scalar + 1.0;
  }
  for (i_matrix_1 = 1; i_matrix_1 <= m; i_matrix_1 += 1) {
    for (i_matrix_0 = 1; i_matrix_0 <= n; i_matrix_0 += 1) {
      matrix[i_matrix_1 - 1][i_matrix_0 - 1] = *scalar + 2.;
    }
  }
  for (i_v2_0 = 1; i_v2_0 <= n; i_v2_0 += 1) {
    v2[i_v2_0 - 1] = matrix[2 - 1][i_v2_0 - 1];
  }
  v2[1 - 1] = 1.;
  return 0;
}
