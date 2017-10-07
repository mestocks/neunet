#ifndef nn_hash_h__
#define nn_hash_h__
#endif

struct nnArgStore {
  FILE *fp;
  char *cmd;
  char *arch;
  struct nnHashTable *arghash;
};

extern void nn_arg_parse(struct nnArgStore *Pmers, int argc, char **argv);
extern void nn_process_activation(struct NeuNet *nnet, char *acts);
extern void nn_get_arch(unsigned long *nnodes, char *arch, unsigned long nlayers);


