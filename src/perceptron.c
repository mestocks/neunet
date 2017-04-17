#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rwk_parse.h>
#include <neurons.h>
#include <algo.h>

#define ARG_NINPUTS 2
#define ARG_NHIDDEN 3
#define ARG_WEIGHTS 4

#define ARG_NET_INDEX 2
  /*

    i    j

    [] \
    [] x []
    [] x []
    [] /

    delta_j = o_k * (1 - o_k) * (d_k - o_k)

    delta_i = h_k * (1 - h_k) * sum(w_ij * delta_j)
   */

// perceptron learn <net.arch> <wts.file>
// perceptron solve <net.arch> <wts.file>

/*
 *
 * echo "0.8 0.2 0.4 0.9 0.3 0.5 0.3 0.5 0.9" > start_weights.txt
 * perceptron learn 2,3,1 start_weights.txt < training_data.txt > learned_weights.txt
 * echo "1 0" | perceptron solve 2,3,1 learned_weights.txt
 *
 */

int main(int argc, char **argv)
{
  int i, j, l;

  int nlayers;
  int *nnodes;
  
  int net_cols;
  char **net_array;
  char net_delim = ',';
  
  net_cols = rwk_countcols(argv[ARG_NET_INDEX], &net_delim);
  nlayers = net_cols - 1;
  
  net_array = calloc(net_cols, sizeof (char*));
  nnodes = calloc(net_cols, sizeof (int));
  rwk_str2array(net_array, argv[ARG_NET_INDEX], net_cols, &net_delim);
  for (i = 0; i < net_cols; i++) {
    nnodes[i] = atoi(net_array[i]);
  }
  
  //
  
  double inputs[nnodes[0]];
  double targets[nnodes[2]];
  
  struct Layer *curr_layer;
  struct Layer *head_layer;
  struct Layer *tail_layer;

  head_layer = calloc(1, sizeof (struct Layer));
  init_layer(head_layer, nnodes[0], nnodes[1]);
  curr_layer = head_layer;
  for (i = 1; i < nlayers; i++) {
    curr_layer->next = calloc(1, sizeof (struct Layer));
    init_layer(curr_layer->next, nnodes[i], nnodes[i + 1]);
    curr_layer->next->prev = curr_layer;
    curr_layer = curr_layer->next;
  }
  tail_layer = curr_layer;



  FILE *wfp;
  wfp = fopen(argv[3], "r");
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

  i = 0;
  curr_layer = head_layer;
  while (curr_layer != NULL) {
    for (j = 0; j < curr_layer->n_input_weights; j++) {
      curr_layer->input_weights[j] = wfloats[i];
      i++;
    }
    curr_layer = curr_layer->next;
  }
  
  free(warray);
  free(wfloats);

  //

  FILE *fp;;
  fp = stdin;
  char **array;
  char buffer[1280000];
  char delim = ' ';
  
  if (strcmp(argv[1], "solve") == 0) {
    array = calloc(nnodes[0], sizeof (char *));
    while (fgets(buffer, sizeof(buffer), fp)) {
      rwk_str2array(array, buffer, nnodes[0], &delim);
      for (i = 0; i < nnodes[0]; i++) {
	inputs[i] = atof(array[i]);
      }
      feed_forward(head_layer, &inputs[0], nnodes[0]);
      printf("%f", tail_layer->hidden[0]);
      for (i = 1; i < tail_layer->nhidden; i++) {
	printf(" %f", tail_layer->hidden[i]);
      }
      printf("\n");
    }
    free(array);
  } else {
    array = calloc(nnodes[0] + nnodes[nlayers - 1], sizeof (char *));
    while (fgets(buffer, sizeof(buffer), fp)) {
      rwk_str2array(array, buffer, nnodes[0] + nnodes[nlayers - 1], &delim);
      for (i = 0; i < nnodes[0]; i++) {
	inputs[i] = atof(array[i]);
      }
      for (i = 0; i < nnodes[nlayers]; i++) {
	targets[i] = atof(array[nnodes[0] + i]);
      }
      feed_forward(head_layer, &inputs[0], nnodes[0]);
      back_propogate(tail_layer, &inputs[0], &targets[0], 0.1);
    } 
    
    curr_layer = head_layer;
    while (curr_layer != NULL) {
      for (i = 0; i < curr_layer->n_input_weights; i++) {
	printf("%f\n", curr_layer->input_weights[i]);
      }
      curr_layer = curr_layer->next;
    }
    free(array);
  }

  free_layers(head_layer);
  
  return 0;
}
