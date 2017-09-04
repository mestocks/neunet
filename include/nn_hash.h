#ifndef nn_hash_h__
#define nn_hash_h__

struct nnHashNode {
  char *key;
  char *value;
  struct nnHashNode *next;
};

struct nnHashTable {
  unsigned long size;
  struct nnHashNode **table;
};

extern unsigned long nn_FNV1a(char *key);
extern unsigned long nn_index_hash(char *key, unsigned long size);

extern void nn_create_hash(struct nnHashTable *hash, unsigned long size);
extern void nn_free_hash(struct nnHashTable *hash);
extern void nn_print_hash(struct nnHashTable *hash);
extern void nn_insert_hash(struct nnHashTable *hash, char *key, char *value);
extern char *nn_lookup_hash(struct nnHashTable *hash, char *key);
extern void nn_print_hash(struct nnHashTable *hash);

#endif
