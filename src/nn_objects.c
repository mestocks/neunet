#include <stdio.h>
#include <stdlib.h>

#include <lar_objects.h>
#include <lar_init.h>

#include <nn_matrix.h>
#include <nn_objects.h>

void create_neunet(struct NeuNet *nnet,
		   unsigned long *nnodes,
		   unsigned long nlayers,
		   unsigned long nobs,
		   unsigned long nbatches)
{
  double *ptr;
  unsigned long n, size;
  
  nnet->nobs = nobs;
  nnet->nlayers = nlayers;
  nnet->nbatches = nbatches;
  nnet->nweights = nlayers - 1;

  nnet->bias_wts = calloc(nlayers - 1, sizeof *nnet->bias_wts);
  nnet->layers = calloc(nlayers, sizeof *nnet->layers);
  nnet->deltas = calloc(nlayers - 1, sizeof *nnet->deltas);
  nnet->bias_deltas = calloc(nlayers - 1, sizeof *nnet->deltas);
  
  // Input layer - do not allocate double array
  n = 0;
  create_smatrix(&nnet->layers[n], nbatches, nnodes[n]);

  // Training output
  create_smatrix(&nnet->output, nbatches, nnodes[n]);
  
  // Hidden layers
  for (n = 1; n < nlayers - 1; n++) {
    ptr = calloc(nbatches * nnodes[n], sizeof *ptr);
    create_smatrix(&nnet->layers[n], nbatches, nnodes[n]);
    attach_smatrix(&nnet->layers[n], ptr);

    ptr = calloc(nbatches * nnodes[n], sizeof *ptr);
    create_smatrix(&nnet->deltas[n - 1], nnodes[n], nbatches);
    attach_smatrix(&nnet->deltas[n - 1], ptr);

    ptr = calloc(nbatches, sizeof *ptr);
    create_smatrix(&nnet->bias_deltas[n - 1], nbatches, 1);
    attach_smatrix(&nnet->bias_deltas[n - 1], ptr);
  }

  // Output layer - no bias in final layer
  n = nlayers - 1;
  ptr = calloc(nbatches * nnodes[n], sizeof *ptr);
  create_smatrix(&nnet->layers[n], nbatches, nnodes[n]);
  attach_smatrix(&nnet->layers[n], ptr);

  ptr = calloc(nbatches * (nnodes[n]), sizeof *ptr);
  create_smatrix(&nnet->deltas[n - 1], nnodes[n], nbatches);
  attach_smatrix(&nnet->deltas[n - 1], ptr);

  ptr = calloc(nbatches, sizeof *ptr);
  create_smatrix(&nnet->bias_deltas[n - 1], nbatches, 1);
  attach_smatrix(&nnet->bias_deltas[n - 1], ptr);

  
  nnet->weights = calloc(nnet->nweights, sizeof *nnet->weights);
  nnet->gradient = calloc(nnet->nweights, sizeof *nnet->gradient);
  nnet->tmp_gradient = calloc(nnet->nweights, sizeof *nnet->tmp_gradient);
  
  // Allocate weights
  for (n = 0; n < nnet->nweights; n++) {
    size = nnodes[n] * nnodes[n + 1];

    // weights
    ptr = calloc(size, sizeof *ptr);
    create_smatrix(&nnet->weights[n], nnodes[n], nnodes[n + 1]);
    attach_smatrix(&nnet->weights[n], ptr);

    // gradient
    ptr = calloc(size, sizeof *ptr);
    create_smatrix(&nnet->gradient[n], nnodes[n + 1], nnodes[n]);
    attach_smatrix(&nnet->gradient[n], ptr);

    // tmp_gradient
    ptr = calloc(size, sizeof *ptr);
    create_smatrix(&nnet->tmp_gradient[n], nnodes[n + 1], nnodes[n]);
    attach_smatrix(&nnet->tmp_gradient[n], ptr);
  }
}

void free_neunet(struct NeuNet *nnet)
{
  unsigned long n, i;
  
  for (n = 0; n < nnet->nlayers; n++) {
    free_smatrix(&nnet->layers[n]);
    if (n != 0) {
      free(nnet->layers[n].ptr);
    }
    
    if (n < nnet->nlayers - 1) {
      free_smatrix(&nnet->deltas[n]);
      free(nnet->deltas[n].ptr);

      free_smatrix(&nnet->bias_deltas[n]);
      free(nnet->bias_deltas[n].ptr);
      
      free_smatrix(&nnet->weights[n]);
      free_smatrix(&nnet->gradient[n]);
      free_smatrix(&nnet->tmp_gradient[n]);
      
      free(nnet->weights[n].ptr);
      free(nnet->gradient[n].ptr);
      free(nnet->tmp_gradient[n].ptr);
    }
  }
  free_smatrix(&nnet->output);

  free(nnet->deltas);
  free(nnet->bias_deltas);
  free(nnet->layers);
  free(nnet->weights);
  free(nnet->gradient);
  free(nnet->tmp_gradient);
}


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
