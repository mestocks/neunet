#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include <rwk_parse.h>
#include <rwk_htable.h>

#include <lar_objects.h>
#include <lar_init.h>

#include <nn_objects.h>
#include <nn_algo.h>
#include <nn_fileIO.h>
#include <nn_metrics.h>

// synapse learn <arch> [--weights=r0,1 --nepochs=1] [file.train]
// synapse solve <arch> [--weights=r0,1] [file.test]

#define ARG_COMM 1
#define ARG_ARCH 2
#define ARG_DEFS "synapse <cmd> <arch> --weights=r0,1 --nepochs=1 --bsize=1 --reg=0 --lambda=1.0 --lrate=1.0"

#define MAX_WTS_RSIZE 5120


void print_output(struct NeuralNetwork *nnet)
{
  lar_printf(" ", "\n", nnet->layers[nnet->nlayers - 1]->T);
}


void print_weights(struct NeuralNetwork *nnet)
{
  int l;
  
  for (l = 0; l < nnet->nlayers - 1; l++) {
    lar_printf("\n", "\n", nnet->bias_wts[l]);
    lar_printf("\n", "\n", nnet->weights[l]);
  }  
}

double r_urange(double min, double max)
{
  double range, div;

  range = (max - min);
  div = RAND_MAX / range;
  return min + (rand() / div);
}

void load_weights(char *fname, struct NeuralNetwork *nnet)
{
  int i, j, l;
  
  if (strcmp(fname, "r0,1") == 0) {
    srand(time(NULL));
    double min, max, scalar;
    for (l = 0; l < nnet->nlayers - 1; l++) {
      scalar = sqrt(nnet->layers[l]->nrows + 1);
      min = -1 / scalar;
      max = 1 / scalar;
      for (i = 0; i < nnet->weights[l]->nrows; i++) {
	for (j = 0; j < nnet->weights[l]->ncols; j++) {
	  *nnet->weights[l]->v[i][j] = r_urange(min, max);
	}
      }
      for (i = 0; i < nnet->bias_wts[l]->nrows; i++) {
	for (j = 0; j < nnet->bias_wts[l]->ncols; j++) {
	  *nnet->bias_wts[l]->v[i][j] = r_urange(min, max);
	}
      }
    }
  } else {
    nn_wts_from_file(nnet, fname);
  }
}


void nn_learn(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y, struct rwkHashTable *hash)
{
  /*
   *
   * --updates=k
   *
   *   k = 1
   *        update weights after every observation (stochastic gradient descent)
   *   1 < k < nobs
   *        mini-batch gradient descent
   *   k = nobs
   *        update weights only after all observations (gradient descent)
   */
  int reg;
  int bsize;
  int nepochs;
  double lrate;
  double lambda;

  reg = atoi((char *)rwk_lookup_hash(hash, "--reg"));
  bsize = atoi((char *)rwk_lookup_hash(hash, "--bsize"));
  nepochs = atoi((char *)rwk_lookup_hash(hash, "--nepochs"));
  lrate = atof((char *)rwk_lookup_hash(hash, "--lrate"));
  lambda = atof((char *)rwk_lookup_hash(hash, "--lambda"));

  int rnum;
  int i, j;
  int epoch_num;
  double cost;
  double error;

  cost = 0.0;
  error = 0.0;
  epoch_num = 0;
  srand(time(NULL));

  int nobs;
  nobs = trdata->nobs;
  
  //cost = nn_cost(trdata, nnet, y);
  error = nn_error(trdata, nnet, y);
  fprintf(stderr, "%d %f\n", epoch_num, error);
  while (1) {

    for (i = 0; i < bsize; i++) {
      rnum = r_urange(0, nobs);
      for (j = 0; j < nnet->ninputs; j++) {
	*(nnet->layers[0]->v[j][0]) = *(trdata->inputs.v[rnum][j]);
      }
      for (j = 0; j < nnet->noutputs; j++) {
	*y->v[j][0] = *(trdata->outputs.v[rnum][j]);
      }
      nn_feed_forward(nnet);
      nn_back_propagation(nnet, y);
    }
    nn_update_weights(nnet, nobs, lambda, reg, lrate);
    //cost = nn_cost(trdata, nnet, y);
    error = nn_error(trdata, nnet, y);
    epoch_num++;
    fprintf(stderr, "%d %f\n", epoch_num, error);

    if (epoch_num == nepochs) {
      print_weights(nnet);
      break;
    }
  }  
}

void nn_solve(FILE *fp, struct NeuralNetwork *nnet)
{
  int i;
  char **array;
  char delim = ' ';
  char buffer[51200];
  
  array = calloc(nnet->ninputs, sizeof (char*));
  while (fgets(buffer, sizeof(buffer), fp)) {
    rwk_str2array(array, buffer, nnet->ninputs, &delim);
    
    for (i = 0; i < nnet->ninputs; i++) {
      *(nnet->layers[0])->v[i][0] = atof(array[i]);
    }
    nn_feed_forward(nnet);
    print_output(nnet);
  }
  free(array);  
}


void args2hash(struct rwkHashTable *hash, int argc_wo_fname, char **argv)
{
  int c;
  int ddash;
  char *tmp;
  char *key;
  char *value;
  
  c = 3;
  while (c < argc_wo_fname) {
    if (argv[c][0] == '-' && argv[c][1] == '-') {
      ddash = 0;
      tmp = &argv[c][0];
      while (*tmp) {
	if (*tmp == '=') {
	  *tmp = '\0';
	  key = malloc(128 * sizeof (char));
	  value = malloc(128 * sizeof (char));
	  strcpy(key, argv[c]);
	  strcpy(value, tmp + 1);
	  rwk_insert_hash(hash, key, value);
	  ddash = 1;
	}
	tmp++;
      }
      if (ddash == 0) {
	key = malloc(128 * sizeof (char));
	value = malloc(128 * sizeof (char));
	strcpy(key, argv[c]);
	value[0] = '1';
	value[1] = '\0';
	rwk_insert_hash(hash, key, value);
      }
      c++;
    } else {
      if (c >= argc_wo_fname - 1 || *argv[c+1] == '-') {
	key = malloc(128 * sizeof (char));
	value = malloc(128 * sizeof (char));
	strcpy(key, argv[c]);
	value[0] = '1';
	value[1] = '\0';
	rwk_insert_hash(hash, key, value);
	c++;
      } else {
	key = malloc(128 * sizeof (char));
	value = malloc(128 * sizeof (char));
	strcpy(key, argv[c]);
	strcpy(value, argv[c + 1]);
	rwk_insert_hash(hash, key, value);
	c+=2;
      }
    }
  }
}

int main(int argc, char **argv)
{
  int i;
  FILE *fp;
  int argc_wo_fname;
  struct rwkHashTable arghash;
  
  if (access(argv[argc - 1], F_OK) != -1) {
    fp = fopen(argv[argc - 1], "r");
    argc_wo_fname = argc - 1;
  } else {
    fp = stdin;
    argc_wo_fname = argc;
  }
  
  int defc;
  char **defv;
  char defaults[1024];
  sprintf(defaults, ARG_DEFS);
  defc = rwk_countcols(defaults, " ");
  defv = calloc(defc, sizeof(char *));
  rwk_str2array(defv, defaults, defc, " ");

  rwk_create_hash(&arghash, 128);
  args2hash(&arghash, defc, defv);
  args2hash(&arghash, argc_wo_fname, argv);
  

  int nlayers;
  int *nnodes;
  char **net_array;
  char net_delim = ',';
  nlayers = rwk_countcols(argv[ARG_ARCH], &net_delim);
  nnodes = calloc(nlayers, sizeof (int));
  net_array = calloc(nlayers, sizeof (char*));
  rwk_str2array(net_array, argv[ARG_ARCH], nlayers, &net_delim);
  for (i = 0; i < nlayers; i++) {
    nnodes[i] = atoi(net_array[i]);
  }
  free(net_array);
  

  char delim = ' ';
  struct lar_matrix y;
  struct NeuralNetwork nnet;
  struct TrainingData trdata;
  struct lar_matrix trinput;
  struct lar_matrix troutput;
  create_network(&nnet, nlayers, &nnodes[0]);
  lar_create_matrix(&y, nnodes[nlayers - 1], 1);
  load_weights((char *)rwk_lookup_hash(&arghash, "--weights"), &nnet);
  
  if (strcmp(argv[ARG_COMM], "solve") == 0) {
    
    nn_solve(fp, &nnet);
    
  } else if (strcmp(argv[ARG_COMM], "learn") == 0) {
    
    nn_file2array(&trdata, fp, nnet.ninputs, nnet.noutputs, &delim);
    //nn_load_trdata(&trinput, &troutput, nnet.ninputs, nnet.noutputs, fp, &delim);
    nn_learn(&trdata, &nnet, &y, &arghash);
    
    lar_free_matrix(&trdata.inputs);
    lar_free_matrix(&trdata.outputs);
    
  } else {
    printf("help\n");
  }
  
  lar_free_matrix(&y);
  free_network(&nnet);
  free(nnodes);
  rwk_free_hash(&arghash);
  fclose(fp);
  
  return 0;
}
