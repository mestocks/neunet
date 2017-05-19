#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include <rwk_parse.h>

#include <lar_objects.h>
#include <lar_init.h>

#include <nn_objects.h>
#include <nn_algo.h>

// artnet learn <arch> <bsize> <nepochs>
// artnet learn 2,3,1 20 1000

#define ARG_COMM 1
#define ARG_ARCH 2
#define ARG_BSIZE 3
#define ARG_NEPOCHS 4
#define ARG_WFILE 5
#define ARG_FILE 6

void nn_learn(char *fname, struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int h, i, j;
  
  long size;
  struct stat st;
  int buffer_size;
  
  stat(fname, &st);
  size = st.st_size;
  buffer_size = 2048 * ((size / 2048) + 1);
  
  FILE *fp;
  fp = fopen(fname, "r");
  
  char *buffer;
  char delim = ' ';
  
  buffer = calloc(buffer_size, sizeof (char));
  fread(buffer, 1, size, fp);
  
  char *tmp;
  tmp = buffer;
  i = j = 0;
  for (h = 0; h < buffer_size; h++) {
    if (buffer[h] == '\n') {
      buffer[h] = '\0';
      
      if (j < nnet->ninputs) {
	*(nnet->layers[0]->v[j][0]) = atof(tmp);
      } else {
	*y->v[j - nnet->ninputs][0] = atof(tmp);
      }
      
      nn_feed_forward(nnet);
      nn_back_propagation(nnet, y);
      
      tmp = &buffer[h + 1];
      i++;
      j = 0;
    } else if (buffer[h] == delim) {
      buffer[h] = '\0';
      if (j < nnet->ninputs) {
	*(nnet->layers[0]->v[j][0]) = atof(tmp);
      } else {
	*y->v[j - nnet->ninputs][0] = atof(tmp);
      }
      tmp = &buffer[h + 1];
      j++;
    }
  }
  
  free(buffer);
  fclose(fp);
}


int main(int argc, char **argv)
{
  int b, h, i, j, l;
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

  int bsize;
  int nepochs;
  bsize = atoi(argv[ARG_BSIZE]);
  nepochs = atoi(argv[ARG_NEPOCHS]);

  struct lar_matrix y;
  struct NeuralNetwork nnet;
  create_network(&nnet, nlayers, &nnodes[0]);
  lar_create_matrix(&y, nnodes[nlayers - 1], 1);

  FILE *wfp;
  char **warray;
  char wdelim = ' ';
  char wbuffer[1028];
  b = i = j = l = 0;
  wfp = fopen(argv[ARG_WFILE], "r");
  warray = calloc(1, sizeof (char*));
  while (fgets(wbuffer, sizeof(wbuffer), wfp)) {
    rwk_str2array(warray, wbuffer, 1, &wdelim);
    if (b < nnet.bias_wts[l]->nrows) {
      *(nnet.bias_wts[l])->v[b][0] = atof(warray[0]);
      b++;
    } else {
      *(nnet.weights[l])->v[i][j] = atof(warray[0]);
      j++;
    }
    if (j == nnet.weights[l]->ncols) {
      j = 0;
      i++;
      if (i == nnet.weights[l]->nrows) {
	i = 0;
	b = 0;
	l++;
      }
    }
  }
  free(warray);
  fclose(wfp);


  FILE *fp;
  char **array;
  char delim = ' ';
  char buffer[5120];

  fp = stdin;
  if (strcmp(argv[ARG_COMM], "solve") == 0) {
    array = calloc(nnet.ninputs, sizeof (char*));
    while (fgets(buffer, sizeof(buffer), fp)) {
      rwk_str2array(array, buffer, nnet.ninputs, &delim);
      
      for (i = 0; i < nnet.ninputs; i++) {
	*(nnet.layers[0])->v[i][0] = atof(array[i]);
      }
      nn_feed_forward(&nnet);
      printf("%f", *(nnet.layers[nlayers - 1])->v[0][0]);
      for (i = 1; i < nnet.noutputs; i++) {
	printf(" %f", *(nnet.layers[nlayers - 1])->v[i][0]);
      }
      printf("\n");
    }
    free(array);
  } else if (strcmp(argv[ARG_COMM], "learn") == 0) {

    nn_learn(argv[ARG_FILE], &nnet, &y);
    
    for (l = 0; l < nnet.nlayers - 1; l++) {

      for (i = 0; i < nnet.bias_wts[l]->nrows; i++) {
	for (j = 0; j < nnet.bias_wts[l]->ncols; j++) {
	  printf("%f\n", *nnet.bias_wts[l]->v[i][j]);
	}
      }
      
      for (i = 0; i < nnet.weights[l]->nrows; i++) {
	for (j = 0; j < nnet.weights[l]->ncols; j++) {
	  printf("%f\n", *nnet.weights[l]->v[i][j]);
	}
      }
    }
    
  } else {
    printf("help\n");
  }

  
  lar_free_matrix(&y);
  free_network(&nnet);
  free(nnodes);
  
  return 0;
}
