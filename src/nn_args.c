#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nn_hash.h>
#include <nn_args.h>

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
