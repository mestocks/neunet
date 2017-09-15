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

  /*
  assert(A->ncols == B->nrows);
  assert(M->nrows == A->nrows);
  assert(M->ncols == B->ncols);
  */
  //printf("%lux%lu %lux%lu %lux%lu\n", M->nrows, M->ncols, A->nrows, A->ncols, B->nrows, B->ncols);
  for (i = 0; i < A->nrows; i++) {
    for (j = 0; j < B->ncols; j++) {
      tmp = 0;
      for (k = 0; k < A->ncols; k++) {
	tmp += A->data[i][k] * B->data[k][j];
      }
      M->data[i][j] = tmp;
    }
  }
  printf("\n");
}

/***************/

struct ShadowMatrix *transpose(struct ShadowMatrix *M)
{
  int i, j, n;
  struct ShadowMatrix *MT;

  MT = calloc(1, sizeof *MT);
  MT->T = M;
  MT->nrows = M->ncols;
  MT->ncols = M->nrows;

  MT->pdata = calloc(MT->nrows, sizeof *MT->pdata);
  for (n = 0; n < MT->nrows; n++) {
    MT->pdata[n] = calloc(MT->ncols, sizeof *MT->pdata);
  }  
  
  for (i = 0; i < MT->nrows; i++) {
    for (j = 0; j < MT->ncols; j++) {
      MT->pdata[i][j] = M->pdata[j][i];
    }
  }
  return MT;
}

void create_shadow_matrix(struct ShadowMatrix *M, int nrows, int ncols)
{
  int n;
  M->nrows = nrows;
  M->ncols = ncols;
  M->pdata = calloc(M->nrows, sizeof *M->pdata);
  for (n = 0; n < M->nrows; n++) {
    M->pdata[n] = calloc(M->ncols, sizeof *M->pdata);
  }
  //M->T = transpose(M);
}

void free_shadow_matrix(struct ShadowMatrix *M)
{
  int i;

  for (i = 0; i < M->nrows; i++) {
    free(M->pdata[i]);
  }
  free(M->pdata);

  for (i = 0; i < M->T->nrows; i++) {
    free(M->T->pdata[i]);
  }
  free(M->T->pdata);
}

void matrix_multiply_naive(struct ShadowMatrix *M, struct ShadowMatrix *A, struct ShadowMatrix *B)
{
  int i, j, k;
  double tmp;
  
  for (i = 0; i < A->nrows; i++) {
    for (j = 0; j < B->ncols; j++) {
      tmp = 0;
      for (k = 0; k < A->ncols; k++) {
	tmp += *A->pdata[i][k] * *B->pdata[k][j];
      }
      *M->pdata[i][j] = tmp;
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
      printf("%s%f", sep, *M->pdata[i][j]);
      sep[0] = ' ';
    }
    printf("\n");
    sep[0] = '\0';
  }
}
