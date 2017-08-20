#ifndef nn_objects_h__
#define nn_objects_h__
#endif

extern double nn_cost(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y);
extern double nn_error(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y);
