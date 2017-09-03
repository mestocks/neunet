#include <stdlib.h>

unsigned long nn_nchar(const char *buffer, const char *delim)
{
  const char *tmp;
  unsigned long nchar;

  nchar = 0;
  tmp = buffer;
  while (*tmp) {
    if (*delim == *tmp) {
      nchar++;
    }
    tmp++;
  }
  return nchar;
}

/*
 * nn_str2array - replace delim with '/0' and insert (char *) into array
 *
 *               +-    +-        +-
 * buffer: ["str1\0str2\0... strn\0"] 
 *           ^     ^         ^
 * array:  [ ptr*, ptr*, ... ptr* ]
 *
 * Note that buffer is not const (delim is replaced with '/0'). No memory is 
 * allocated (char pointers from buffer are used in array). This also 
 * means that the pointers stored in array are freed when buffer is freed.
 *
 * if (ndel + 1 > ncols) returns -1
 * if (ndel <= ncols) NULL pointer added to array indices ndel to ncols - 1 & returns 0
 *
 * Example usage:
 *
 * char **array = calloc(ncols, sizeof (char *));
 * if (nn_str2array(array, buffer, ncols, &delim) == -1) {
 *   free(array);
 *   // free buffer if necessary
 *   exit(1);
 * } else {
 *   // do something with array
 * }
 * free(array);
 * // free buffer if necessary
 */
int nn_str2array(char **array, char *buffer, const unsigned long ncols, const char *delim)
{
  char *tmp;
  char newline;
  unsigned long coln;

  coln = 0;
  tmp = buffer;
  newline = '\n';
  
  array[coln] = tmp;
  while (*tmp && newline != *tmp) {
    
    if (*delim == *tmp) {
      *tmp = '\0';
      coln++;
      tmp++;
      if (coln == ncols) {
	return -1;
      } 
      array[coln] = tmp;
    } else {
      tmp++;
    }
  }
  *tmp = '\0';
  
  while (coln < ncols - 1) {
    coln++;
    array[coln] = NULL;
  }

  return 0;
}
