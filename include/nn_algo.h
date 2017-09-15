#ifndef lar_objects_h__
#define lar_objects_h__
#endif

#ifndef nn_matrix_h__
#define nn_matrix_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

#ifndef nn_algo_h__
#define nn_algo_h__

extern void minibatch_feed_forward(struct NeuNet *nnet);
extern void minibatch_back_propagation(struct NeuNet *nnet);
extern void minibatch_update_weights(struct NeuNet *nnet, unsigned long m, double lambda, int reg, double lrate);
extern void batch_vec_feed_forward(struct NeuralNetwork *nnet);
extern void nn_feed_forward(struct NeuralNetwork *nnet);
extern void nn_back_propagation(struct NeuralNetwork *nnet, struct lar_matrix *y);
extern void nn_update_weights(struct NeuralNetwork *nnet, int m, double lambda, int reg, double lrate);

#endif
