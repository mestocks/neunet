#ifndef nn_objects_h__
#define nn_objects_h__
#endif

#ifndef nn_matrix_h__
#define nn_matrix_h__
#endif

#ifndef nn_fileIO_h__
#define nn_fileIO_h__

extern void nn_wts_from_file(struct NeuNet *nnet, char *fname);

extern void nn_file2array(struct InOutData *iodata, FILE *fp, int ninputs, int noutputs, char *delim);

#endif
