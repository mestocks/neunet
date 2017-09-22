#ifndef nn_matrix_h__
#define nn_matrix_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

struct NeuNet {
  unsigned long ninputs;
  unsigned long noutputs;
  unsigned long nlayers;
  unsigned long nweights;
  unsigned long nbatches;
  double *bias_wts;
  struct SMatrix output;
  struct SMatrix *deltas;
  struct SMatrix *bias_deltas;
  struct SMatrix *gradient;
  struct SMatrix *tmp_gradient;
  struct SMatrix *layers;
  struct SMatrix *weights;

  double (**acts) (double);
  double (**dacts) (double);
};

extern void create_neunet(struct NeuNet *nnet,
			  unsigned long *nnodes,
			  unsigned long nlayers,
			  unsigned long nbatches);

extern void free_neunet(struct NeuNet *nnet);


struct InOutData {
  double *input_data;
  double *output_data;
  struct SMatrix inputs;
  struct SMatrix outputs;
};
