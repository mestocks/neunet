#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <lar_objects.h>
#include <lar_init.h>
#include <lar_algebra.h>

#include <nn_objects.h>

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dsigmoid(x) (x * (1-x))

/*
 *  
 * 
 *
 */

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


void nn_back_propagation(struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int i, j, l;
  int nlayers;

  struct lar_matrix *x;
  struct lar_matrix *a;
  struct lar_matrix *b;
  struct lar_matrix *d;
  struct lar_matrix *dplus;
  struct lar_matrix *D;
  struct lar_matrix *W;

  nlayers = nnet->nlayers;
  // deltas
  for (l = nlayers - 1; l > 0; l--) {
    d = nnet->deltas[l - 1];
    D = nnet->gradient[l - 1];
    x = nnet->layers[l - 1];
    a = nnet->layers[l];

    if (l == nlayers - 1) {
      for (i = 0; i < a->nrows; i++) {
	*d->v[i][0] = (*y->v[i][0] - *a->v[i][0]) * (*a->v[i][0] * (1 - *a->v[i][0]));
      }
    } else {
      W = nnet->weights[l];
      dplus = nnet->deltas[l];
      lar_matrix_multiply_naive(d, W, dplus);
      
      for (i = 0; i < d->nrows; i++) {
	for (j = 0; j < d->ncols; j++) {
	  *d->v[i][j] = *d->v[i][j] * (*a->v[i][j] * (1 - *a->v[i][j]));
	}
      }
    }
    lar_matrix_multiply_naive(D, x, d->T);
  }

  // update weights
  for (l = nlayers - 1; l > 0; l--) {
    b = nnet->bias_wts[l - 1];
    d = nnet->deltas[l - 1];
    D = nnet->gradient[l - 1];
    W = nnet->weights[l - 1];
    for (i = 0; i < W->nrows; i++) {
      for (j = 0; j < W->ncols; j++) {
	*W->v[i][j] += *D->v[i][j];
      }
    }
    for (i = 0; i < b->nrows; i++) {
      for (j = 0; j < b->ncols; j++) {
	*b->v[i][j] += *d->v[i][j];
      }
    }
  }
}
