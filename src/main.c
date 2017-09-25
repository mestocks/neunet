#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nn_matrix.h>
#include <nn_objects.h>
#include <nn_hash.h>
#include <nn_algo.h>
#include <nn_args.h>
#include <nn_fileIO.h>
#include <nn_metrics.h>
#include <nn_string.h>

#define ARG_DEFS "neunet <cmd> <arch> --weights=rSQRT --nepochs=1 --bsize=1 --reg=0 --lambda=1.0 --lrate=1.0 [file]"

#define MAX_WTS_RSIZE 5120

double r_urange(double min, double max)
{
  double range, div;

  range = (max - min);
  div = RAND_MAX / range;
  return min + (rand() / div);
}

void print_weights(struct NeuNet *nnet)
{
  unsigned long l, i, j;
  
  for (l = 0; l < nnet->nweights; l++) {
    printf("%f\n", nnet->bias_wts[l]);
    for (i = 0; i < nnet->weights[l].nrows; i++) {
      for (j = 0; j < nnet->weights[l].ncols; j++) {
	printf("%f\n", nnet->weights[l].data[i][j]);
      }
    }
  }
}


void load_weights(char *fname, struct NeuNet *nnet)
{
  int i, j, l;
  
  if (strcmp(fname, "rSQRT") == 0) {
    srand(time(NULL));
    double min, max, scalar;
    for (l = 0; l < nnet->nlayers - 1; l++) {
      scalar = sqrt(nnet->layers[l].nrows + 1);
      min = -1 / scalar;
      max = 1 / scalar;
      for (i = 0; i < nnet->weights[l].nrows; i++) {
	for (j = 0; j < nnet->weights[l].ncols; j++) {
	  nnet->weights[l].data[i][j] = r_urange(min, max);
	}
      }
      nnet->bias_wts[l] = r_urange(min, max);
    }
  } else {
    nn_wts_from_file(nnet, fname);
  }
}


void nn_learn(struct NeuNet *nnet, struct SMatrix *inputs, struct SMatrix *outputs, struct nnArgStore *Pmers)
{
  char *e;
  int base;
  int reg;
  //unsigned long bsize;
  unsigned long nepochs;
  double lrate;
  double lambda;

  base = 10;
  
  reg = atoi(nn_lookup_hash(Pmers->arghash, "reg"));
  //bsize = atoi(nn_lookup_hash(Pmers->arghash, "bsize"));
  nepochs = strtoul(nn_lookup_hash(Pmers->arghash, "nepochs"), &e, base);
  lrate = atof(nn_lookup_hash(Pmers->arghash, "lrate"));
  lambda = atof(nn_lookup_hash(Pmers->arghash, "lambda"));

  int rnum;
  unsigned long i, j;
  unsigned long epoch_num;
  double cost;
  double error;

  cost = 0.0;
  error = 0.0;
  epoch_num = 0;
  srand(time(NULL));

  unsigned long nobs;
  nobs = inputs->nrows;
  
  error = nn_error(nnet, inputs, outputs);
  fprintf(stderr, "%lu %f\n", epoch_num, error);
  
  while (1) {

    for (i = 0; i < nnet->nbatches; i++) {
      rnum = r_urange(0, nobs);
      nnet->layers[0].data[i] = inputs->data[rnum];
      nnet->output.data[i] = outputs->data[rnum];
    }
    minibatch_feed_forward(nnet);
    minibatch_back_propagation(nnet);
    // nobs passed to update as I think that it should be lambda / nobs
    // rather than the number of batches (even though this doesn't make
    // much sense)
    minibatch_update_weights(nnet, nobs, lambda, reg, lrate);
    
    error = nn_error(nnet, inputs, outputs);
    epoch_num++;
    fprintf(stderr, "%lu %f\n", epoch_num, error);

    if (epoch_num == nepochs) {
      print_weights(nnet);
      break;
    }
  }  
}

void nn_solve(FILE *fp, struct NeuNet *nnet)
{
  unsigned long i;
  char **array;
  char delim = ' ';
  char buffer[51200];
  double *inputs;
  unsigned long bsize;

  bsize = 1;
  inputs = calloc(bsize * nnet->ninputs, sizeof *inputs);
  
  array = calloc(nnet->ninputs, sizeof (char*));
  while (fgets(buffer, sizeof(buffer), fp)) {
    nn_str2array(array, buffer, nnet->ninputs, &delim);
    
    for (i = 0; i < nnet->ninputs; i++) {
      inputs[i] = atof(array[i]);
    }
    
    nnet->layers[0].data[0] = inputs;
    minibatch_feed_forward(nnet);
    
    printf("%f", nnet->layers[nnet->nlayers - 1].data[0][0]);
    for (i = 1; i < nnet->noutputs; i++) {
      printf(" %f", nnet->layers[nnet->nlayers - 1].data[0][i]);
    }
    printf("\n");
  }
  
  free(array);
  free(inputs);
  
}


int main(int argc, char **argv)
{
  int i;
  char *e;
  int base;
  char **defv;
  unsigned long defc;
  char defaults[1024];
  struct nnArgStore *Pmers;

  base = 10;
  
  sprintf(defaults, ARG_DEFS);
  defc = nn_nchar(defaults, " ") + 1;
  defv = calloc(defc, sizeof *defv);
  nn_str2array(defv, defaults, defc, " ");

  Pmers = calloc(1, sizeof *Pmers);
  Pmers->arghash = calloc(1, sizeof *Pmers->arghash);
  nn_create_hash(Pmers->arghash, 128);
  
  nn_arg_parse(Pmers, (int)defc, defv);
  nn_arg_parse(Pmers, argc, argv);
  
  unsigned long n;
  unsigned long nlayers;
  unsigned long *nnodes;
  char **net_array;
  char net_delim = ',';

  nlayers = (unsigned long)nn_nchar(Pmers->arch, &net_delim) + 1;
  nnodes = calloc(nlayers, sizeof *nnodes);
  net_array = calloc(nlayers, sizeof *net_array);
  nn_str2array(net_array, Pmers->arch, nlayers, &net_delim);
  for (n = 0; n < nlayers; n++) {
    nnodes[n] = atoi(net_array[n]);
  }
  free(net_array);
  
  char delim = ' ';
  unsigned long bsize;
  struct NeuNet neunet;

  bsize = strtoul(nn_lookup_hash(Pmers->arghash, "bsize"), &e, base);
  create_neunet(&neunet, nnodes, nlayers, bsize);
  load_weights(nn_lookup_hash(Pmers->arghash, "weights"), &neunet);

  if (strcmp(Pmers->cmd, "solve") == 0) {
    
    nn_solve(Pmers->fp, &neunet);
    
  } else if (strcmp(Pmers->cmd, "learn") == 0) {
    
    struct InOutData iodata;

    nn_file2array(&iodata, Pmers->fp, nnodes[0], nnodes[nlayers - 1], &delim);
    nn_learn(&neunet, &iodata.inputs, &iodata.outputs, Pmers);

    free_smatrix(&iodata.inputs);
    free_smatrix(&iodata.outputs);
    free(iodata.input_data);
    free(iodata.output_data);
    
  } else {
    printf("help\n");
  }
  
  free_neunet(&neunet);
  free(nnodes);

  nn_free_hash(Pmers->arghash);
  free(Pmers->arghash);
  fclose(Pmers->fp);
  free(Pmers);
  free(defv);
  
  return 0;
}
