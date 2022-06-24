#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_logical_statements_c(int v1, int v2, int *v_xor, int *v_xnor, int *v_nand, 
  int *v_neqv, int * restrict v_v_val) {

  /* Array casts for pointer arguments */
  int (*v_val) = (int (*)) v_v_val;
  
  *v_xor = v1 && !v2 || !v1 && v2;
  *v_xnor = v1 && v2 || !(v1 || v2);
  *v_nand = !(v1 && v2);
  *v_neqv = !(v1 && v2) && (v1 || v2);
  v_val[1 - 1] = true;
  v_val[2 - 1] = false;
  
  return 0;
}
