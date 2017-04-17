#include <stdlib.h>
#include <neurons.h>

void init_layer(struct Layer *layer, int ninputs, int nhidden)
{
  layer->ninputs = ninputs;
  layer->nhidden = nhidden;
  layer->n_input_weights = ninputs * nhidden;
  
  layer->input_weights = calloc(layer->n_input_weights, sizeof (double));
  layer->hidden = calloc(layer->nhidden, sizeof (double));
  layer->tmp_h = calloc(layer->nhidden, sizeof (double));
  layer->deltas = calloc(layer->nhidden, sizeof (double));
  layer->wdelta_sum = calloc(layer->ninputs, sizeof (double));
  
  layer->next = NULL;
  layer->prev = NULL;
}

void free_layer(struct Layer *layer)
{
  free(layer->input_weights);
  free(layer->hidden);
  free(layer->tmp_h);
  free(layer->deltas);
  free(layer->wdelta_sum);
}


void free_layers(struct Layer *head_layer)
{
  struct Layer *curr_layer;

  while (head_layer != NULL) {
    curr_layer = head_layer;
    head_layer = curr_layer->next;
    free_layer(curr_layer);
    free(curr_layer);
  }
}
