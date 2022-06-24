#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
int transpile_multibody_conditionals_c(int in1, int *out1, int *out2) {

  
  if (in1 > 5) {
    *out1 = 5;
  } else {
    *out1 = 1;
  }
  
  if (in1 < 0) {
    *out2 = 0;
  } else if (in1 > 5) {
    *out2 = 6;
    *out2 = *out2 - 1;
  } else if (3 < in1 && in1 <= 5) {
    *out2 = 4;
  } else {
    *out2 = in1;
  }
  return 0;
}
