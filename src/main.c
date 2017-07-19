#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/stat.h>

#include <rwk_parse.h>
#include <rwk_htable.h>

#include <lar_objects.h>
#include <lar_init.h>

#include <nn_objects.h>
#include <nn_algo.h>

// synapse learn <arch> [--weights=r0,1 --nepochs=1] [file.train]
// synapse solve <arch> [--weights=r0,1] [file.test]

#define ARG_COMM 1
#define ARG_ARCH 2
#define ARG_DEFS "synapse <cmd> <arch> --weights=r0,1 --nepochs=1"

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
    FILE *fp;
    char **array;
    char delim = ' ';
    char buffer[1028];
    int b;
    
    b = i = j = l = 0;
    fp = fopen(fname, "r");
    array = calloc(1, sizeof (char*));
    while (fgets(buffer, sizeof(buffer), fp)) {
      rwk_str2array(array, buffer, 1, &delim);
      if (b < nnet->bias_wts[l]->nrows) {
	*(nnet->bias_wts[l])->v[b][0] = atof(array[0]);
	b++;
      } else {
	*(nnet->weights[l])->v[i][j] = atof(array[0]);
	j++;
      }
      if (j == nnet->weights[l]->ncols) {
	j = 0;
	i++;
	if (i == nnet->weights[l]->nrows) {
	  i = 0;
	  b = 0;
	  l++;
      }
      }
    }
    free(array);
    fclose(fp);  
  }
}

struct TrainingData {
  int nobs;
  int ninputs;
  int noutputs;
  double **inputs;
  double **outputs;
};

void file2array(struct TrainingData *trdata, FILE *fp, int ninputs, int noutputs, char *delim)
{
  int h, i, j, n;  
  long size;
  char *tmp;
  char *last;
  char *buffer;
  struct stat st;
  int buffer_size;
  
  fstat(fileno(fp), &st);
  size = st.st_size;
  buffer_size = 2048 * ((size / 2048) + 1);
  buffer = calloc(buffer_size, sizeof (char));
  fread(buffer, 1, size, fp);

  n = 0;
  tmp = buffer;
  while (*tmp) {
    if (*tmp == '\n') {
      n++;
    }
    tmp++;
  }
  trdata->nobs = n;
  trdata->ninputs = ninputs;
  trdata->noutputs = noutputs;
  trdata->inputs = calloc(trdata->nobs, sizeof (double *));
  trdata->outputs = calloc(trdata->nobs, sizeof (double *));

  for (n = 0; n < trdata->nobs; n++) {
    trdata->inputs[n] = calloc(trdata->ninputs, sizeof (double));
    trdata->outputs[n] = calloc(trdata->noutputs, sizeof (double));
  }
  
  i = j = 0;
  last = tmp = buffer;
  while (*tmp) {
    if (*tmp == ' ') {
      *tmp = '\0';
      if (j < trdata->ninputs) {
	trdata->inputs[i][j] = atof(last);
      } else {
	trdata->outputs[i][j - trdata->ninputs] = atof(last);
      }
      j++;
      tmp++;
      last = tmp;
    } else if (*tmp == '\n') {
      *tmp = '\0';
      if (j < trdata->ninputs) {
	trdata->inputs[i][j] = atof(last);
      } else {
	trdata->outputs[i][j - trdata->ninputs] = atof(last);
      }
      j = 0;
      i++;
      tmp++;
      last = tmp;
    } else {
      tmp++;
    }
  }
  free(buffer);
}


void nn_learn(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int i, j;
  
  i = j = 0;
  for (i = 0; i < trdata->nobs; i++) {
    for (j = 0; j < trdata->ninputs; j++) {
      *(nnet->layers[0]->v[j][0]) = trdata->inputs[i][j];
    }
    for (j = 0; j < trdata->noutputs; j++) {
      *y->v[j][0] = trdata->outputs[i][j];
    }
    
    nn_feed_forward(nnet);
    nn_back_propagation(nnet, y);
    nn_update_weights(nnet);
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
  struct rwkHashTable arghash;
  rwk_create_hash(&arghash, 128);

  FILE *fp;
  int argc_wo_fname;
  
  if (access(argv[argc - 1], F_OK) != -1) {
    fp = fopen(argv[argc - 1], "r");
    argc_wo_fname = argc - 1;
  } else {
    fp = stdin;
    argc_wo_fname = argc;
  }
  
  int defc;
  char **defv;
  char defaults[128];

  sprintf(defaults, ARG_DEFS);
  defc = rwk_countcols(defaults, " ");
  defv = calloc(defc, sizeof(char *));
  rwk_str2array(defv, defaults, defc, " ");
  
  args2hash(&arghash, defc, defv);
  args2hash(&arghash, argc_wo_fname, argv);
  
  int i;
  int nlayers;
  char net_delim = ',';
  nlayers = rwk_countcols(argv[ARG_ARCH], &net_delim);
  
  int *nnodes;
  char **net_array;
  nnodes = calloc(nlayers, sizeof (int));
  net_array = calloc(nlayers, sizeof (char*));
  rwk_str2array(net_array, argv[ARG_ARCH], nlayers, &net_delim);
  for (i = 0; i < nlayers; i++) {
    nnodes[i] = atoi(net_array[i]);
  }
  free(net_array);
  
  int nepochs;
  nepochs = atoi((char *)rwk_lookup_hash(&arghash, "--nepochs"));

  struct lar_matrix y;
  struct NeuralNetwork nnet;
  create_network(&nnet, nlayers, &nnodes[0]);
  lar_create_matrix(&y, nnodes[nlayers - 1], 1);
  int n;
  char delim = ' ';
  
  load_weights((char *)rwk_lookup_hash(&arghash, "--weights"), &nnet);
  
  if (strcmp(argv[ARG_COMM], "solve") == 0) {
    nn_solve(fp, &nnet);
  } else if (strcmp(argv[ARG_COMM], "learn") == 0) {
    struct TrainingData trdata;
    
    file2array(&trdata, fp, nnet.ninputs, nnet.noutputs, &delim);
    for (i = 0; i < nepochs; i++) {
      nn_learn(&trdata, &nnet, &y);
    }
    print_weights(&nnet);
    
    for (n = 0; n < trdata.nobs; n++) {
      free(trdata.inputs[n]);
      free(trdata.outputs[n]);
    }
    free(trdata.inputs);
    free(trdata.outputs);
    
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
