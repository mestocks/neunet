#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <nn_matrix.h>
#include <nn_objects.h>
#include <nn_hash.h>
#include <nn_args.h>
#include <nn_string.h>
#include <nn_algo.h>

void nn_arg_parse(struct nnArgStore *Pmers, int argc, char **argv)
{
  int c;
  char *tmp;
  int argc_wo_fname;

  c = 1;
  Pmers->cmd = argv[c];
  c++;
  Pmers->arch = argv[c];
  c++;

  if (access(argv[argc - 1], F_OK) != -1) {
    Pmers->fp = fopen(argv[argc - 1], "r");
    argc_wo_fname = argc - 1;
  } else {
    Pmers->fp = stdin;
    argc_wo_fname = argc;
  }
  
  while (c < argc_wo_fname) {
    if (argv[c][0] == '-') {
      if (argv[c][1] == '-') {
	tmp = &argv[c][2];
	while (*tmp) {
	  if (*tmp == '=') {
	    *tmp = '\0';
	    nn_insert_hash(Pmers->arghash, argv[c] + 2, tmp + 1);
	  }
	  tmp++;
	}
      } else {
	// single dash commands
      }
    }
    c++;
  }
}


void nn_get_arch(unsigned long *nnodes, char *arch, unsigned long nlayers)
{
  unsigned long n;
  char **net_array;
  char arch_cpy[1028];
  char net_delim = ',';

  strcpy(arch_cpy, arch);
  net_array = calloc(nlayers, sizeof *net_array);
  nn_str2array(net_array, arch_cpy, nlayers, &net_delim);
  for (n = 0; n < nlayers; n++) {
    nnodes[n] = atoi(net_array[n]);
  }
  free(net_array);
}


void nn_print_args(struct nnArgStore *Pmers)
{
  unsigned long i;
  struct nnHashNode *curr;

  printf("arch %s\n", Pmers->arch);
  for (i = 0; i < Pmers->arghash->size; i++) {
    curr = Pmers->arghash->table[i];
    while (curr != NULL) {
      printf("%s %s\n", curr->key, curr->value);
      curr = curr->next;
    }
  }
}

void nn_process_activation(struct NeuNet *nnet, char *acts)
{
  char *act;
  char **act_array;
  char delim = ' ';
  unsigned long n, nacts;
  double (*pact) (double x);
  double (*dpact) (double x);
  
  nacts = nn_nchar(acts, ",") + 1;
  act_array = calloc(nacts, sizeof *act_array);
  nn_str2array(act_array, acts, nacts, ",");

  for (n = 0; n < nnet->nlayers - 1; n++) {
    
    if (nacts == 1) {
      act = act_array[0];
    } else {
      act = act_array[n];
    }

    if (strcmp(act, "sigmoid") == 0) {
      pact = nn_sigmoid;
      dpact = nn_dsigmoid;
    } else if (strcmp(act, "tanh") == 0) {
      pact = nn_tanh;
      dpact = nn_dtanh;
    } else if (strcmp(act, "ReLU") == 0) {
      pact = nn_ReLU;
      dpact = nn_dReLU;
    } else if (strcmp(act, "lReLU") == 0) {
      pact = nn_lReLU;
      dpact = nn_dlReLU;
    }
    
    nnet->acts[n] = pact;
    nnet->dacts[n] = dpact;
  }
  free(act_array);
}
