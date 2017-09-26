#include <stdio.h>
#include <stdlib.h>

#include <nn_matrix.h>
#include <nn_objects.h>

void create_neunet(struct NeuNet *nnet,
		   unsigned long *nnodes,
		   unsigned long nlayers,
		   //		   unsigned long nobs,
		   unsigned long nbatches)
{
  double *ptr;
  unsigned long n, size;
  
  //  nnet->nobs = nobs;
  nnet->nlayers = nlayers;
  nnet->nbatches = nbatches;
  nnet->nweights = nlayers - 1;
  nnet->ninputs = nnodes[0];
  nnet->noutputs = nnodes[nlayers - 1];
  
  //nnet->bias_wts = calloc(nlayers - 1, sizeof *nnet->bias_wts);
  nnet->layers = calloc(nlayers, sizeof *nnet->layers);
  nnet->deltas = calloc(nlayers - 1, sizeof *nnet->deltas);
  nnet->bias_deltas = calloc(nlayers - 1, sizeof *nnet->bias_deltas);
  
  // Input & output layer - do not allocate double array
  n = 0;
  create_smatrix(&nnet->layers[n], nbatches, nnodes[n]);
  create_smatrix(&nnet->output, nbatches, nnodes[nlayers - 1]);

  
  // Hidden layers
  for (n = 1; n < nlayers; n++) {
    ptr = calloc(nbatches * nnodes[n], sizeof *ptr);
    create_smatrix(&nnet->layers[n], nbatches, nnodes[n]);
    attach_smatrix(&nnet->layers[n], ptr);

    ptr = calloc(nbatches * nnodes[n], sizeof *ptr);
    create_smatrix(&nnet->deltas[n - 1], nnodes[n], nbatches);
    attach_smatrix(&nnet->deltas[n - 1], ptr);

    ptr = calloc(nnodes[n], sizeof *ptr);
    create_smatrix(&nnet->bias_deltas[n - 1], nnodes[n], 1);
    attach_smatrix(&nnet->bias_deltas[n - 1], ptr);
    
  }

  nnet->bias_wts = calloc(nlayers - 1, sizeof *nnet->bias_wts);
  nnet->weights = calloc(nnet->nweights, sizeof *nnet->weights);
  nnet->gradient = calloc(nnet->nweights, sizeof *nnet->gradient);
  
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

    // bias weights
    ptr = calloc(nnodes[n + 1], sizeof *ptr);
    create_smatrix(&nnet->bias_wts[n], 1, nnodes[n + 1]);
    attach_smatrix(&nnet->bias_wts[n], ptr);
  }
  nnet->acts = calloc(nlayers - 1, sizeof *nnet->acts);
  nnet->dacts = calloc(nlayers - 1, sizeof *nnet->dacts);
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
      free(nnet->weights[n].ptr);
      
      free_smatrix(&nnet->gradient[n]);
      free(nnet->gradient[n].ptr);

      free_smatrix(&nnet->bias_wts[n]);
      free(nnet->bias_wts[n].ptr);
    }
  }
  free_smatrix(&nnet->output);

  free(nnet->deltas);
  free(nnet->bias_wts);
  free(nnet->bias_deltas);
  free(nnet->layers);
  free(nnet->weights);
  free(nnet->gradient);

  free(nnet->acts);
  free(nnet->dacts);
}
