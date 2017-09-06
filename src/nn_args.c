#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nn_hash.h>
#include <nn_args.h>
//#include <rwk_htable.h>

struct nnArgStore *nn_arg_parse(int argc, char **argv)
{
  int c;
  char *tmp;
  int argc_wo_fname;
  struct nnArgStore *Pmers;

  Pmers = calloc(1, sizeof *Pmers);
  Pmers->arghash = calloc(1, sizeof *Pmers->arghash);
  nn_create_hash(Pmers->arghash, 128);

  c = 1;
  Pmers->cmd = argv[c];
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
	c++;
      } else {
	// single dash commands
	//c+=2;
      }
    }
  }

  return Pmers;
}

/*
void nn_args2hash(struct rwkHashTable *hash, int argc_wo_fname, char **argv)
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
*/
