#include <stdio.h>
#include <stdlib.h>

#include <lar_objects.h>
#include <lar_init.h>

#include <nn_objects.h>


void create_network(struct NeuralNetwork *nnet, int nlayers, int *nnodes)
{
  int i, j, k;
  
  nnet->nlayers = nlayers;
  nnet->ninputs = nnodes[0];
  nnet->noutputs = nnodes[nlayers - 1];

  nnet->nhidden = calloc(nlayers - 2, sizeof (int));
  for (i = 1; i < nlayers - 1; i++) {
    nnet->nhidden[i - 1] = nnodes[1];
  }

  nnet->layers = calloc(nlayers, sizeof (struct lar_matrix *));
  nnet->weights = calloc(nlayers - 1, sizeof (struct lar_matrix *));
  nnet->deltas = calloc(nlayers - 1, sizeof (struct lar_matrix *));
  nnet->gradient = calloc(nlayers - 1, sizeof (struct lar_matrix *));
  nnet->tmp_gradient = calloc(nlayers - 1, sizeof (struct lar_matrix *));
  nnet->bias_wts = calloc(nlayers - 1, sizeof (struct lar_matrix *));

  for (i = 0; i < nlayers; i++) {
      nnet->layers[i] = calloc(1, sizeof (struct lar_matrix));
      lar_create_matrix(nnet->layers[i], nnodes[i], 1);
    if (i < nlayers - 1) {
      nnet->deltas[i] = calloc(1, sizeof (struct lar_matrix));
      nnet->weights[i] = calloc(1, sizeof (struct lar_matrix));
      nnet->gradient[i] = calloc(1, sizeof (struct lar_matrix));
      nnet->tmp_gradient[i] = calloc(1, sizeof (struct lar_matrix));
      nnet->bias_wts[i] = calloc(1, sizeof (struct lar_matrix));
      lar_create_matrix(nnet->deltas[i], nnodes[i + 1], 1);
      lar_create_matrix(nnet->weights[i], nnodes[i], nnodes[i + 1]);
      lar_create_matrix(nnet->gradient[i], nnodes[i], nnodes[i + 1]);
      lar_create_matrix(nnet->tmp_gradient[i], nnodes[i], nnodes[i + 1]);
      lar_create_matrix(nnet->bias_wts[i], nnodes[i + 1], 1);
      for (j = 0; j < nnet->gradient[i]->nrows; j++) {
	for (k = 0; k < nnet->gradient[i]->ncols; k++) {
	  *nnet->gradient[i]->v[j][k] = 0.0;
	}
      }
    }
  }
}

void free_network(struct NeuralNetwork *nnet)
{
  int i;

  for (i = 0; i < nnet->nlayers; i++) {
    if (i < nnet->nlayers - 1) {
      lar_free_matrix(nnet->bias_wts[i]);
      lar_free_matrix(nnet->gradient[i]);
      lar_free_matrix(nnet->tmp_gradient[i]);
      lar_free_matrix(nnet->weights[i]);
      lar_free_matrix(nnet->deltas[i]);
      free(nnet->bias_wts[i]);
      free(nnet->gradient[i]);
      free(nnet->tmp_gradient[i]);
      free(nnet->weights[i]);
      free(nnet->deltas[i]);
    }
    lar_free_matrix(nnet->layers[i]);
    free(nnet->layers[i]);
  }
  free(nnet->bias_wts);
  free(nnet->gradient);
  free(nnet->tmp_gradient);
  free(nnet->deltas);
  free(nnet->weights);
  free(nnet->layers);
  free(nnet->nhidden);
}
