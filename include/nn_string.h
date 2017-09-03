#ifndef nn_string_h__
#define nn_string_h__

extern unsigned long nn_nchar(const char *buffer, const char *delim);
extern int nn_str2array(char **array, char *buffer, const unsigned long ncols, const char *delim);

#endif
