#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <nn_matrix.h>


void create_smatrix(struct SMatrix *M, unsigned long nrows, unsigned long ncols)
{
  M->nrows = nrows;
  M->ncols = ncols;
  M->data = calloc(M->nrows, sizeof *M->data);
}


void attach_smatrix(struct SMatrix *M, double *data)
{
  unsigned long i;
  
  M->ptr = data;
  for (i = 0; i < M->nrows; i++) {
    M->data[i] = &data[i * M->ncols];
  }
}


void print_smatrix(struct SMatrix *M)
{
  char sep[2];
  unsigned long i, j;

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

void free_smatrix(struct SMatrix *M)
{
  free(M->data);
}


void smatrix_multiply(struct SMatrix *M, struct SMatrix *A, struct SMatrix *B)
{
  double tmp;
  unsigned long i, j, k;

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
