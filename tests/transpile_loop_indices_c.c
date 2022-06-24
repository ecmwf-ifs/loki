#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_loop_indices_c(int n, int idx, int * restrict v_mask1, 
  int * restrict v_mask2, double * restrict v_mask3) {
  
  int i;
  /* Array casts for pointer arguments */
  int (*mask1) = (int (*)) v_mask1;
  int (*mask2) = (int (*)) v_mask2;
  double (*mask3) = (double (*)) v_mask3;
  
  for (i = 1; i <= n; i += 1) {
    if (i < idx) {
      mask1[i - 1] = 1;
    }
    
    if (i == idx) {
      mask1[i - 1] = 2;
    }
    
    mask2[i - 1] = i;
  }
  mask3[n - 1] = 3.0;
  return 0;
}
