#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_intrinsics_c(double v1, double v2, double v3, double v4, double *vmin, 
  double *vmax, double *vabs, double *vmin_nested, double *vmax_nested) {

  
  *vmin = fmin(v1, v2);
  *vmax = fmax(v1, v2);
  *vabs = fabs(v1 - v2);
  *vmin_nested = fmin(fmin(v1, v2), fmin(v3, v4));
  *vmax_nested = fmax(fmax(v1, v2), fmax(v3, v4));
  return 0;
}
