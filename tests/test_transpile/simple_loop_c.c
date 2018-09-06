#include <stdio.h>

int simple_loop_c(int n, int m, double v_scalar, double *v_vector, double *v_tensor) {
  int i, j;

  double (*vector) = (double *) v_vector;
  double (*tensor)[n] = (double (*)[n]) v_tensor;
  
  printf("C::n, m => %d, %d\n", n, m);
  printf("C::scalar %lf\n", v_scalar);

  printf("C::vector ");
  for (i=0; i<n; i++) printf("%lf, ", vector[i]);
  printf("\n");

  for (j=0; j<m; j++) {
    printf("C::tensor ");
    for (i=0; i<n; i++) printf("%lf, ", tensor[j][i]);
    printf("\n");
  }

  /* Now, we do the actual work equivalent to the F-tester */
  for (i=0; i<n; i++) {
    vector[i] = vector[i] + tensor[1][i] + 1.0;
  }

  // Note: Data is column-major, so traverse accordingly
  for (j=0; j<m; j++) {
    for (i=0; i<n; i++) {
      tensor[j][i] = 10.*(double)(j+1) + (double)(i+1);
    }
  }

  printf("\nAfter some work...\n", n, m);
  for (j=0; j<m; j++) {
    printf("C::tensor ");
    for (i=0; i<n; i++) printf("%lf, ", tensor[j][i]);
    printf("\n");
  }

  return 0;
}
