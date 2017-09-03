#ifndef lar_objects_h__
#define lar_objects_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

/*
 *
 * nlayers = input_layer
 *         + n_hidden_layers
 *         + output_layer;
 *
 * layers = [input, hidden1, ..., output]
 *
 * weights = [input:hidden1, ..., hiddenN:output]
 *
 */

struct NeuralNetwork {
  int nlayers;
  int ninputs;
  int noutputs;
  int *nhidden;
  
  struct lar_matrix **layers;
  struct lar_matrix **weights;
  struct lar_matrix **deltas;
  struct lar_matrix **gradient;
  struct lar_matrix **tmp_gradient;
  struct lar_matrix **bias_wts;
};

extern void create_network(struct NeuralNetwork *nnet, int nlayers, int *nnodes);
extern void free_network(struct NeuralNetwork *nnet);

struct TrainingData {
  unsigned long nobs;
  int ninputs;
  int noutputs;
  struct lar_matrix inputs;
  struct lar_matrix outputs;
};
