#ifndef nn_matrix_h__
#define nn_matrix_h__


struct SMatrix {
  unsigned long nrows;
  unsigned long ncols;
  double *ptr;
  double **data;
};

extern void create_smatrix(struct SMatrix *M, unsigned long nrows, unsigned long ncols);
extern void attach_smatrix(struct SMatrix *M, double *data);
extern void print_smatrix(struct SMatrix *M);
extern void free_smatrix(struct SMatrix *M);
extern void smatrix_multiply(struct SMatrix *M, struct SMatrix *A, struct SMatrix *B);

/*****************/

struct ShadowMatrix {
  int nrows;
  int ncols;
  double ***pdata;
  struct ShadowMatrix *T;
};

extern struct ShadowMatrix *transpose(struct ShadowMatrix *M);
extern void create_shadow_matrix(struct ShadowMatrix *M, int nrows, int ncols);
extern void free_shadow_matrix(struct ShadowMatrix *M);
extern void matrix_multiply_naive(struct ShadowMatrix *M, struct ShadowMatrix *A, struct ShadowMatrix *B);
extern void print_matrix(struct ShadowMatrix *M);

#endif
