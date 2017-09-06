#ifndef nn_hash_h__
#define nn_hash_h__
#endif

struct nnArgStore {
  FILE *fp;
  char *cmd;
  char *arch;
  struct nnHashTable *arghash;
};

void nn_arg_parse(struct nnArgStore *Pmers, int argc, char **argv);



