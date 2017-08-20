#ifndef nn_objects_h__
#define nn_objects_h__
#endif

#ifndef nn_fileIO_h__
#define nn_fileIO_h__

extern void nn_wts_from_file(struct NeuralNetwork *nnet, char *fname);
extern void nn_load_trdata(struct lar_matrix *trinput, struct lar_matrix *troutput, int ninputs, int noutputs, FILE *fp, char *delim);
extern void nn_file2array(struct TrainingData *trdata, FILE *fp, int ninputs, int noutputs, char *delim);

#endif
