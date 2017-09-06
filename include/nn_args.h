//#ifndef rwk_htable_h__
//#define rwk_htable_h__
//#endif

#ifndef nn_hash_h__
#define nn_hash_h__
#endif

struct nnArgStore {
  FILE *fp;
  char *cmd;
  struct nnHashTable *arghash;
};

extern struct nnArgStore *nn_arg_parse(int argc, char **argv);
//extern void nn_args2hash(struct rwkHashTable *hash, int argc_wo_fname, char **argv);



