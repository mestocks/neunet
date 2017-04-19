#ifndef neurons_h__
#define neurons_h__

struct Layer {
  int ninputs;
  int nhidden;
  int n_input_weights;

  // ptr->[ptr, ...]
  //        v
  //      [double, 
  
  double **weights;
  
  double *hidden;
  double *tmp_h;
  double *deltas;
  double *wdelta_sum;
};

extern void init_layer(struct Layer *layer, int ninputs, int nhidden);
extern void free_layer(struct Layer *layer);
extern void free_layers(struct Layer *head_layer, int nlayers);

#endif
