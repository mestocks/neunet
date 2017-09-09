#include <stdio.h>
#include <stdlib.h>
#include <nn_matrix.h>

void create_shadow_matrix(struct ShadowMatrix *M, int nrows, int ncols)
{
  int i;
  M->nrows = nrows;
  M->ncols = ncols;
  M->data = calloc(nrows, sizeof *M->data);
  for (i = 0; i < nrows; i++) {
    M->data[i] = calloc(ncols, sizeof *M->data[i]);
  }
}

void matrix_multiply_naive(struct ShadowMatrix *M, struct ShadowMatrix *A, struct ShadowMatrix *B)
{
  int i, j, k;
  double tmp;
  
  for (i = 0; i < A->nrows; i++) {
    for (j = 0; j < B->ncols; j++) {
      tmp = 0;
      for (k = 0; k < A->ncols; k++) {
	tmp += A->data[i][k] * B->data[k][j];
      }
      M->data[i][j] = tmp;
    }
  }
}

void print_matrix(struct ShadowMatrix *M)
{
  int i, j;
  char sep[2];

  sep[0] = '\0';
  sep[1] = '\0';
  
  for (i = 0; i < M->nrows; i++) {
    for (j = 0; j < M->ncols; j++) {
      printf("%s%f", sep, M->data[i][j]);
      sep[0] = ' ';
    }
    printf("\n");
    sep[0] = '\0';
  }
}
