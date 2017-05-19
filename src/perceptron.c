#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/stat.h>

#include <rwk_parse.h>
#include <neurons.h>
#include <algo.h>

#define ARG_COMMAND 1
#define ARG_NET_ARCH 2
#define ARG_WTSFILE 3
#define ARG_OPTS_START 4

  /*

    i    j

    [] \
    [] x []
    [] x []
    [] /

    delta_j = o_k * (1 - o_k) * (d_k - o_k)

    delta_i = h_k * (1 - h_k) * sum(w_ij * delta_j)
   */

// perceptron learn <net.arch> <wts.file> [--stochastic-gradient-descent] <input.file>
// perceptron solve <net.arch> <wts.file> <input.file>

/*
 *
 * echo "0.8 0.2 0.4 0.9 0.3 0.5 0.3 0.5 0.9" > start_weights.txt
 * perceptron learn 2,3,1 start_weights.txt < training_data.txt > learned_weights.txt
 * echo "1 0" | perceptron solve 2,3,1 learned_weights.txt
 *
 */

void print_hidden(struct Layer *layer, int nlayers)
{
  int l, h;
  char sep[2];

  sep[0] = '\0';
  sep[1] = '\0';

  for (l = 0; l < nlayers; l++) {
    for (h = 0; h < layer[l].nhidden; h++) {
      printf("%s%f", sep, layer[l].hidden[h]);
      sep[0] = ' ';
    }
  }
    printf("\n");
}

void print_weights(struct Layer *layer, int nlayers)
{
  int l, h, i;
  
  for (l = 0; l < nlayers; l++) {
    for (i = 0; i < layer[l].ninputs; i++) {
      for (h = 0; h < layer[l].nhidden; h++) {
	printf("%f\n", layer[l].weights[i][h]);
      }
    }
  }
}
      
int main(int argc, char **argv)
{
  int h, i, j, l, r;

  int nlayers;
  int *nnodes;
  
  int net_cols;
  char **net_array;
  char net_delim = ',';
  
  net_cols = rwk_countcols(argv[ARG_NET_ARCH], &net_delim);
  nlayers = net_cols - 1;
  
  net_array = calloc(net_cols, sizeof (char*));
  nnodes = calloc(net_cols, sizeof (int));
  rwk_str2array(net_array, argv[ARG_NET_ARCH], net_cols, &net_delim);
  for (i = 0; i < net_cols; i++) {
    nnodes[i] = atoi(net_array[i]);
  }

  double *inputs;
  double *targets;

  struct Layer *head_layer;

  head_layer = calloc(nlayers, sizeof (struct Layer));
  for (l = 0; l < nlayers; l++) {
    init_layer(&head_layer[l], nnodes[l], nnodes[l + 1]);
  }
  

  FILE *wfp;
  wfp = fopen(argv[ARG_WTSFILE], "r");
  int nweights;
  char **warray;
  double *wfloats;
  char wbuffer[1028];
  char wdelim = ' ';
  nweights = 0;
  for (i = 0; i < nlayers; i++) {
    nweights += nnodes[i] * nnodes[i + 1];
  }
  wfloats = calloc(nweights, sizeof (double));
  i=0;
  while (fgets(wbuffer, sizeof(wbuffer), wfp)) {
    warray = calloc(1, sizeof (char*));
    rwk_str2array(warray, wbuffer, 1, &wdelim);
    wfloats[i] = atof(warray[0]);
    i++;
  }

  j = 0;
  for (l = 0; l < nlayers; l++) {
    for (i = 0; i < head_layer[l].ninputs; i++) {
      for (h = 0; h < head_layer[l].nhidden; h++) {
	head_layer[l].weights[i][h] = wfloats[j];
	j++;
      }
    }
  }
  free(warray);
  free(wfloats);

  //
  
  if (strcmp(argv[ARG_COMMAND], "solve") == 0) {

    FILE *fp;
    fp = fopen(argv[argc-1], "r");
    
    char **array;
    char buffer[1280000];
    char delim = ' ';

    inputs = calloc(nnodes[0], sizeof (double));
    
    array = calloc(nnodes[0], sizeof (char *));
    while (fgets(buffer, sizeof(buffer), fp)) {
      rwk_str2array(array, buffer, nnodes[0], &delim);
      for (i = 0; i < nnodes[0]; i++) {
	inputs[i] = atof(array[i]);
      }
      feed_forward(head_layer, nlayers, &inputs[0], nnodes[0]);
      
      printf("%f", head_layer[nlayers - 1].hidden[0]);
      for (i = 1; i < head_layer[nlayers - 1].nhidden; i++) {
	printf(" %f", head_layer[nlayers - 1].hidden[i]);
      }
      printf("\n");
    }
    free(array);
    free(inputs);
    fclose(fp);
    
  } else if (strcmp(argv[ARG_COMMAND], "learn") == 0) {

    if (strcmp(argv[ARG_OPTS_START], "--stochastic-gradient-descent") == 0) {

      long size;
      struct stat st;
      int buffer_size;

      stat(argv[argc-1], &st);
      size = st.st_size;
      buffer_size = 2048 * ((size / 2048) + 1);

      FILE *fp;
      char *buffer;
      char delim = ' ';

      fp = fopen(argv[argc-1], "r");
      buffer = calloc(buffer_size, sizeof (char));
      fread(buffer, 1, size, fp);

      int nrows;
      int ndoubles;

      nrows = 60000;
      ndoubles = nnodes[0] + nnodes[nlayers];
      
      double *training_mem;
      double **training_array;

      training_array = calloc(nrows, sizeof (double*));
      training_mem = calloc(nrows * ndoubles, sizeof (double));
      for (i = 0; i < nrows; i++) {
	training_array[i] = &training_mem[i * ndoubles];
      }

      char *tmp;
      tmp = buffer;
      i = j = 0;
      for (h = 0; h < buffer_size; h++) {
	if (buffer[h] == '\n') {
	  buffer[h] = '\0';
	  training_array[i][j] = atof(tmp);
	  tmp = &buffer[h + 1];
	  i++;
	  j = 0;
	} else if (buffer[h] == ' ') {
	  buffer[h] = '\0';
	  training_array[i][j] = atof(tmp);
	  tmp = &buffer[h + 1];
	  j++;
	}
      }
      fclose(fp);
      free(buffer);

      int last;
      double error;
      double rec_last;
      double diff;

      last = nlayers - 1;

      for (l = 0; l < nlayers; l++) {
	for (h = 0; h < head_layer[l].nhidden; h++) {
	  head_layer[l].deltas[h] = 0.0;
	}
      }

      double lrate;
      double bdelta;
      double brate;
      int batch_size;
      double batch_mean;
      batch_size = 10;
      batch_mean = 1 / (double)batch_size;
      
      lrate = 3.0;
      brate = batch_mean * lrate;
      
      int rnd;
      srand(time(NULL));

      int niter;
      niter = nrows;
      //niter = 3000;
      
      for (r = 0; r < niter; r++) {
	rnd = randint(nrows - 1);
	
	inputs = &training_array[rnd][0];
	targets = &training_array[rnd][nnodes[0]];
	
	feed_forward(head_layer, nlayers, inputs, nnodes[0]);
	
	deltas(head_layer, nlayers, inputs, targets);
	if ((r + 1) % batch_size == 0) {

	  fprintf(stderr, "%d %d", r, rnd);
	  for (h = 0; h < head_layer[last].nhidden; h++) {
	    diff = targets[h] - head_layer[last].hidden[h];
	    fprintf(stderr, " %f", diff * diff);
	  }

	  /*
	   *
	   * [   ]   [ ]
	   * [   ] * [ ] * [   ]
	   * [   ]   [ ]
	   *
	   */

	  for (l = nlayers - 1; l >= 0; l--) {
	    for (h = 0; h < head_layer[l].nhidden; h++) {
	      bdelta = brate * head_layer[l].deltas[h];
	      for (i = 0; i < head_layer[l].ninputs; i++) {
		if (l == 0) {
		  head_layer[l].weights[i][h] += bdelta * inputs[i];
		} else {
		  head_layer[l].weights[i][h] += bdelta * head_layer[l - 1].hidden[i];
		}
	      }
	      head_layer[l].deltas[h] = 0.0;
	    }
	  }

	  fprintf(stderr, "\n");
	}
      }

      print_weights(head_layer, nlayers);
      
      free(training_mem);
      free(training_array);
    } else {
      
      FILE *fp;
      fp = fopen(argv[argc-1], "r");
      
      char **array;
      char buffer[1280000];
      char delim = ' ';

      inputs = calloc(nnodes[0], sizeof (double));
      targets = calloc(nnodes[nlayers], sizeof (double));
      
      array = calloc(nnodes[0] + nnodes[nlayers], sizeof (char *));
      while (fgets(buffer, sizeof(buffer), fp)) {
	rwk_str2array(array, buffer, nnodes[0] + nnodes[nlayers], &delim);
	for (i = 0; i < nnodes[0]; i++) {
	  inputs[i] = atof(array[i]);
	}
	for (i = 0; i < nnodes[nlayers]; i++) {
	  targets[i] = atof(array[nnodes[0] + i]);
	}
	feed_forward(head_layer, nlayers, &inputs[0], nnodes[0]);
	back_propagate(head_layer, nlayers, &inputs[0], &targets[0], 0.1);
      }

      print_weights(head_layer, nlayers);
      
      free(inputs);
      free(targets);
    }
  }
  free_layers(head_layer, nlayers);
  
  return 0;
}
