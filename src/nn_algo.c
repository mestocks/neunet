#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <lar_objects.h>
#include <lar_init.h>
#include <lar_algebra.h>

#include <nn_matrix.h>
#include <nn_objects.h>

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dsigmoid(x) (x * (1-x))

#define tanh(x) ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
// if a == tanh(x) then dtanh(a):
#define dtanh(x) (1 - (x * x))

#define ReLU(x) (x > 0 ? x : 0)
#define dReLU(x) (x < 0 ? 0 : 1)

#define lReLU(x) (x > 0.01 * x ? x : 0.01 * x)
#define dlReLU(x) (x < 0 ? 0.01 : 1)
/*
 *
 *      / nodes \
 * X = [. . . . .] \
 *     [. . . . .]  training examples
 *     [. . . . .] /
 *
 * A = (X * W) + b
 *
 */

void minibatch_feed_forward(struct NeuNet *nnet)
{
  unsigned long i, j, l;
  struct SMatrix *A, *W, *X;
  
  for (l = 0; l < nnet->nlayers - 1; l++) {
    A = &nnet->layers[l + 1];
    W = &nnet->weights[l];
    X = &nnet->layers[l];
    
    smatrix_multiply(A, X, W);
    for (i = 0; i < A->nrows; i++) {
      for (j = 0; j < A->ncols; j++) {
	A->data[i][j] = sigmoid(A->data[i][j] + nnet->bias_wts[l]);
      }
    }
  }
}





void batch_vec_feed_forward(struct NeuralNetwork *nnet)
{
  int i, l;
  struct lar_matrix *A, *W, *X;
  
  for (l = 0; l < nnet->nlayers - 1; l++) {
    A = nnet->layers[l + 1];
    W = nnet->weights[l];
    X = nnet->layers[l];
    for (i = 0; i < X->nrows; i++) {
      *X->v[i][X->ncols - 1] = 1.0;
    }
    lar_matrix_multiply_naive(A, X, W);
  }
}

void feed_forward2(struct NeuralNetwork *nnet)
{
  int l;
  struct lar_matrix *A, *W, *X;
  
  for (l = 0; l < nnet->nlayers - 1; l++) {
    A = nnet->layers[l + 1];
    W = nnet->weights[l];
    X = nnet->layers[l];
    *X->v[0][0] = 1.0; // not correct
    lar_matrix_multiply_naive(A, W->T, X);
  }
}


void nn_feed_forward(struct NeuralNetwork *nnet)
{
  int i, j, l;
  int nlayers;
  struct lar_matrix *x;
  struct lar_matrix *a;
  struct lar_matrix *b;
  struct lar_matrix *W;
  
  nlayers = nnet->nlayers;
  for (l = 0; l < nlayers - 1; l++) {
    x = nnet->layers[l];
    W = nnet->weights[l];
    a = nnet->layers[l + 1];
    b = nnet->bias_wts[l];
    lar_matrix_multiply_naive(a, W->T, x);
    for (i = 0; i < a->nrows; i++) {
      for (j = 0; j < a->ncols; j++) {
	*a->v[i][j] += *b->v[i][j];
    	*a->v[i][j] = sigmoid(*a->v[i][j]);
      }
    }
  }
}

/*
 *  
 * dZ[2] = A[2] - Y
 * dW[2] = (1 / m) * dZ[2] * A[1].T
 * db[2] = (1 / m) * np.sum(dZ[2], axis = 1, keepdims = True)
 * 
 * dZ[1] = W[2].T * dZ[2] .* g[1]' * Z[1]
 * dW[1] = (1 / m) * dZ[1] * X.T
 * db[1] = (1 / m) * np.sum(dZ[1], axis = 1, keepdims = True)
 *
 * W = W - lrate * dW
 * b = b - lrate * db
 */

void minibatch_back_propagation(struct NeuNet *nnet)
{
  unsigned long i, j, l, m;
  unsigned long olayer, nlayers;

  struct SMatrix *W, *D;
  struct SMatrix *A, *AA;
  struct SMatrix *d, *db, *dplus;

  m = nnet->nbatches;
  nlayers = nnet->nlayers;
  olayer = nlayers - 1;

  for (l = olayer; l > 0; l--) {

    A = &nnet->layers[l - 1];
    AA = &nnet->layers[l];
    d = &nnet->deltas[l - 1];
    W = &nnet->weights[l];
    db = &nnet->bias_deltas[l - 1];
    D = &nnet->gradient[l - 1];
    dplus = &nnet->deltas[l];

    if (l == olayer) {      
      for (i = 0; i < AA->nrows; i++) {
	for (j = 0; j < AA->ncols; j++) {
	  d->data[j][i] = AA->data[i][j] - nnet->output.data[i][j];
	}
      }
    } else {
      smatrix_multiply(d, W, dplus);
      for (i = 0; i < d->nrows; i++) {
	for (j = 0; j < d->ncols; j++) {
	  d->data[i][j] = d->data[i][j] * dsigmoid(AA->data[j][i]);	  
	}
      }
    }
    
    smatrix_multiply(D, d, A);
    for (i = 0; i < D->nrows; i++) {
      for (j = 0; j < D->ncols; j++) {
	D->data[i][j] = D->data[i][j] / m;
      }
    }
    
    for (i = 0; i < d->ncols; i++) {
      for (j = 0; j < d->nrows; j++) {
	db->data[i][0] += d->data[j][i];
      }
      db->data[i][0] = db->data[i][0] / m;
    }
  }
}

void minibatch_update_weights(struct NeuNet *nnet,
			      unsigned long m, double lambda, int reg, double lrate)
{
  unsigned long i, j, l;
  unsigned long nlayers;
  
  struct SMatrix *d;
  struct SMatrix *db;
  struct SMatrix *D;
  struct SMatrix *W;

  nlayers = nnet->nlayers;
  for (l = 0; l < nlayers - 1; l++) {
    D = &nnet->gradient[l];
    W = &nnet->weights[l];
    db = &nnet->bias_deltas[l];
    
    if (reg == 0) {
      for (i = 0; i < W->nrows; i++) {
	for (j = 0; j < W->ncols; j++) {
	  W->data[i][j] = W->data[i][j] - (lrate * D->data[j][i]);
	}
      }
    } else {
      for (i = 0; i < W->nrows; i++) {
	for (j = 0; j < W->ncols; j++) {
	  // is this correct?
	  W->data[i][j] = W->data[i][j] - (lrate * (D->data[j][i] + ((lambda / m) * W->data[i][j])));
	}
      }
    }
    
    for (i = 0; i < D->nrows; i++) {
      for (j = 0; j < D->ncols; j++) {
	D->data[i][j] = 0.0;
      }
    }

    for (i = 0; i < db->nrows; i++) {
      for (j = 0; j < db->ncols; j++) {
	db->data[i][j] = 0.0;
      }
    }
    
  }
}


void nn_back_propagation(struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int i, j, l;
  int nlayers;

  struct lar_matrix *x;
  struct lar_matrix *a;
  struct lar_matrix *d;
  struct lar_matrix *dplus;
  struct lar_matrix *D;
  struct lar_matrix *tmpD;
  struct lar_matrix *W;

  nlayers = nnet->nlayers;
  
  for (l = nlayers - 1; l > 0; l--) {
    d = nnet->deltas[l - 1];
    tmpD = nnet->tmp_gradient[l - 1];
    D = nnet->gradient[l - 1];
    x = nnet->layers[l - 1];
    a = nnet->layers[l];

    if (l == nlayers - 1) {
      for (i = 0; i < a->nrows; i++) {
	*d->v[i][0] = (*y->v[i][0] - *a->v[i][0]) * dsigmoid(*a->v[i][0]);
	//*d->v[i][0] = (*a->v[i][0] - *y->v[i][0]) * dsigmoid(*a->v[i][0]);
      }
    } else {
      W = nnet->weights[l];
      dplus = nnet->deltas[l];
      lar_matrix_multiply_naive(d, W, dplus);
      
      for (i = 0; i < d->nrows; i++) {
	for (j = 0; j < d->ncols; j++) {
	  *d->v[i][j] = *d->v[i][j] * dsigmoid(*a->v[i][j]);
	}
      }
    }
    lar_matrix_multiply_naive(tmpD, x, d->T);
    for (i = 0; i < D->nrows; i++) {
      for (j = 0; j < D->ncols; j++) {
	*D->v[i][j] += *tmpD->v[i][j];
      }
    }
  }
}


void nn_update_weights(struct NeuralNetwork *nnet, int m, double lambda, int reg, double lrate)
{
  int i, j, l;
  int nlayers;
  
  struct lar_matrix *b;
  struct lar_matrix *d;
  struct lar_matrix *D;
  struct lar_matrix *W;

  nlayers = nnet->nlayers;
  for (l = nlayers - 1; l > 0; l--) {
    
    b = nnet->bias_wts[l - 1];
    d = nnet->deltas[l - 1];
    D = nnet->gradient[l - 1];
    W = nnet->weights[l - 1];
    
    if (reg == 0) {
      for (i = 0; i < W->nrows; i++) {
	for (j = 0; j < W->ncols; j++) {
	  *W->v[i][j] += lrate * (*D->v[i][j] / m);
	}
      }
    } else {
      for (i = 0; i < W->nrows; i++) {
	for (j = 0; j < W->ncols; j++) {
	  *W->v[i][j] += lrate * ((*D->v[i][j] / m) + ((lambda / m) * *W->v[i][j]));
	}
      }
    }
    
    for (i = 0; i < b->nrows; i++) {
      for (j = 0; j < b->ncols; j++) {
	*b->v[i][j] += lrate * (*d->v[i][j] / m);
      }
    }
    
    for (i = 0; i < D->nrows; i++) {
      for (j = 0; j < D->ncols; j++) {
	*D->v[i][j] = 0.0;
      }
    }
  }
}
