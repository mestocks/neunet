#ifndef nn_matrix_h__
#define nn_matrix_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

extern double nn_cost(struct NeuNet *nnet, struct SMatrix *inputs, struct SMatrix *outputs);
extern double nn_error(struct NeuNet *nnet, struct SMatrix *inputs, struct SMatrix *outputs);
