#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <nn_matrix.h>
#include <nn_objects.h>

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
  double (*pact) (double x);
  struct SMatrix *A, *W, *X;
  
  for (l = 0; l < nnet->nlayers - 1; l++) {
    A = &nnet->layers[l + 1];
    W = &nnet->weights[l];
    X = &nnet->layers[l];
    pact = nnet->acts[l];
    
    smatrix_multiply(A, X, W);
    for (i = 0; i < A->nrows; i++) {
      for (j = 0; j < A->ncols; j++) {
	//A->data[i][j] = sigmoid(A->data[i][j] + nnet->bias_wts[l]);
	A->data[i][j] = pact(A->data[i][j] + nnet->bias_wts[l]);
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

  double (*dpact) (double x);

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
    dpact = nnet->dacts[l - 1];

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
	  //	  d->data[i][j] = d->data[i][j] * dsigmoid(AA->data[j][i]);
	  d->data[i][j] = d->data[i][j] * dpact(AA->data[j][i]);
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
	  // For L2 regularization should be:
	  // dW = dW + ((lamda / m) * W)
	  // W = W - (alpha * dW)
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
