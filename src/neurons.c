#include <stdlib.h>
#include <neurons.h>

void init_layer(struct Layer *layer, int ninputs, int nhidden)
{
  int i;
  
  layer->ninputs = ninputs;
  layer->nhidden = nhidden;
  layer->n_input_weights = ninputs * nhidden;
  
  layer->weights = calloc(layer->ninputs, sizeof (double*));
  for (i = 0; i < layer->ninputs; i++) {
    layer->weights[i] = calloc(layer->nhidden, sizeof (double));
  }

  layer->hidden = calloc(layer->nhidden, sizeof (double));
  layer->tmp_h = calloc(layer->nhidden, sizeof (double));
  layer->deltas = calloc(layer->nhidden, sizeof (double));
  layer->wdelta_sum = calloc(layer->ninputs, sizeof (double));
}

void free_layer(struct Layer *layer)
{
  int i;
  
  free(layer->hidden);
  free(layer->tmp_h);
  free(layer->deltas);
  free(layer->wdelta_sum);

  for (i = 0; i < layer->ninputs; i++) {
    free(layer->weights[i]);
  }
  free(layer->weights);
}


void free_layers(struct Layer *head_layer, int nlayers)
{
  int l;

  for (l = 0; l < nlayers; l++) {
    free_layer(&head_layer[l]);
  }
}
