#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <nn_matrix.h>
#include <nn_objects.h>
#include <nn_algo.h>

double nn_error(struct NeuNet *nnet, struct SMatrix *inputs, struct SMatrix *outputs)
{
  unsigned long b, m, i, j;
  double error;
  double tmp_error;
  double exp;
  double obs;

  b = 0;
  error = 0.0;

  for (m = 0; m < inputs->nrows; m++) {

    nnet->layers[0].data[b] = inputs->data[m];
    nnet->output.data[b] = outputs->data[m];
    
    if (b == nnet->nbatches - 1) {
      minibatch_feed_forward(nnet);
      for (i = 0; i < nnet->output.nrows; i++) {
	for (j = 0; j < nnet->output.ncols; j++) {
	  exp = nnet->output.data[i][j];
	  obs = nnet->layers[nnet->nlayers - 1].data[i][j];
	  tmp_error = obs - exp;
	  error += tmp_error * tmp_error;
	}
      }
      b = 0;
    } else {
      b++;
    }
  }
  return error / (2 * inputs->nrows);    
}

double nn_cost(struct NeuNet *nnet, struct SMatrix *inputs, struct SMatrix *outputs)
{
  unsigned long b, m, i, j;
  double cost;
  double tmp_cost;
  double exp;
  double obs;

  cost = 0.0;

  for (m = 0; m < inputs->nrows; m++) {

    nnet->layers[0].data[b] = inputs->data[m];
    nnet->output.data[b] = outputs->data[m];
    
    if (b == nnet->nbatches - 1) {
      minibatch_feed_forward(nnet);
      for (i = 0; i < nnet->output.nrows; i++) {
	for (j = 0; j < nnet->output.ncols; j++) {
	  exp = nnet->output.data[i][j];
	  obs = nnet->layers[0].data[i][j];
	  tmp_cost = (-exp * log(obs)) - ((1 - exp) * log(1 - obs));
	  cost += tmp_cost;
	}
      }
      b = 0;
    } else {
      b++;
    }
  }
  return cost / inputs->nrows;
}
