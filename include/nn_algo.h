#ifndef lar_objects_h__
#define lar_objects_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

#ifndef nn_algo_h__
#define nn_algo_h__


extern void nn_feed_forward(struct NeuralNetwork *nnet);
extern void nn_back_propagation(struct NeuralNetwork *nnet, struct lar_matrix *y);

#endif
