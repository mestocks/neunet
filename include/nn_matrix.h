#ifndef nn_matrix_h__
#define nn_matrix_h__

struct ShadowMatrix {
  int nrows;
  int ncols;
  double **data;
};

extern void create_shadow_matrix(struct ShadowMatrix *M, int nrows, int ncols);
extern void matrix_multiply_naive(struct ShadowMatrix *M, struct ShadowMatrix *A, struct ShadowMatrix *B);
extern void print_matrix(struct ShadowMatrix *M);

#endif
