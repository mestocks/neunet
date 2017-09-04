#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <nn_hash.h>

const uint64_t FNV_PRIME = 1099511628211;
const uint64_t FNV_OFFSET = 14695981039346656037;


unsigned long nn_FNV1a(char *key)
{  
  unsigned long i;
  unsigned long hash;
  hash = FNV_OFFSET;
  for (i = 0; i < strlen(key); i++) {
    hash ^= key[i];
    hash *= FNV_PRIME;
  }
  return hash;
}


unsigned long nn_index_hash(char *key, unsigned long size)
{
  unsigned long index;
  unsigned long hval;

  hval = nn_FNV1a(key);
  index = hval % size;
  return index;
}


void nn_print_hash(struct nnHashTable *hash)
{
  unsigned long i;
  struct nnHashNode *curr;

  for (i = 0; i < hash->size; i++) {
    printf("%lu", i);
    curr = hash->table[i];
    while (curr != NULL) {
      printf(" %s:%s", curr->key, curr->value);
      curr = curr->next;
    }
    printf("\n");
  }
}

void nn_create_hash(struct nnHashTable *hash, unsigned long hsize)
{
  hash->size = hsize;
  hash->table = calloc(hsize, sizeof *hash->table);
}


void nn_insert_hash(struct nnHashTable *hash, char *key, char *value)
{
  unsigned long index;
  struct nnHashNode *new, *curr, *prev;

  index = nn_index_hash(key, hash->size);
  curr = hash->table[index];

  while (curr != NULL && strcmp(curr->key, key) != 0) {
    prev = curr;
    curr = curr->next;
  }

  if (curr != NULL) {
    curr->value = value;
  } else {
    new = calloc(1, sizeof *new);
    new->key = key;
    new->value = value;
    new->next = hash->table[index];
    hash->table[index] = new;
  }
}


char *nn_lookup_hash(struct nnHashTable *hash, char *key)
{
  char *value;
  unsigned long index;
  struct nnHashNode *curr;

  index = nn_index_hash(key, hash->size);
  curr = hash->table[index];

  value = NULL;
  while (curr != NULL && strcmp(curr->key, key) != 0) {
    curr = curr->next;
  }

  if (curr != NULL) {
    value = curr->value;
  }
  
  return value;
}

void nn_free_hash(struct nnHashTable *hash)
{
  unsigned long i;
  struct nnHashNode *curr, *prev;

  for (i = 0; i < hash->size; i++) {
    curr = hash->table[i];
    while (curr != NULL) {
      prev = curr;
      curr = curr->next;
      free(prev);
    }
  }
  free(hash->table);
}
