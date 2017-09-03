#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
//#include <rwk_parse.h>
#include <lar_objects.h>
#include <lar_init.h>
#include <nn_objects.h>
#include <nn_string.h>

#define MAX_WTS_RSIZE 5120

void nn_wts_from_file(struct NeuralNetwork *nnet, char *fname)
{
  int b, i, j, l;
  FILE *fp;
  char **array;
  char delim = ' ';
  char buffer[MAX_WTS_RSIZE];

  fp = fopen(fname, "r");
  
  b = i = j = l = 0;
  
  array = calloc(1, sizeof (char*));
  while (fgets(buffer, sizeof(buffer), fp)) {
    nn_str2array(array, buffer, 1, &delim);
    if (b < nnet->bias_wts[l]->nrows) {
      *(nnet->bias_wts[l])->v[b][0] = atof(array[0]);
      b++;
    } else {
      *(nnet->weights[l])->v[i][j] = atof(array[0]);
      j++;
    }
    if (j == nnet->weights[l]->ncols) {
      j = 0;
      i++;
      if (i == nnet->weights[l]->nrows) {
	i = 0;
	b = 0;
	l++;
      }
    }
  }
  free(array);
  fclose(fp);  
}


void nn_file2array(struct TrainingData *trdata, FILE *fp, int ninputs, int noutputs, char *delim)
{
  int h, i, j;
  unsigned long n;
  long size;
  char *tmp;
  char *last;
  char *buffer;
  struct stat st;
  int buffer_size;
  
  fstat(fileno(fp), &st);
  size = st.st_size;
  buffer_size = 2048 * ((size / 2048) + 1);
  buffer = calloc(buffer_size, sizeof (char));
  fread(buffer, 1, size, fp);

  /*
  n = 0;
  tmp = buffer;
  while (*tmp) {
    if (*tmp == '\n') {
      n++;
    }
    tmp++;
  }
  */

  n = nn_nchar(buffer, "\n");
  
  trdata->nobs = n;
  trdata->ninputs = ninputs;
  trdata->noutputs = noutputs;
  lar_create_matrix(&trdata->inputs, trdata->nobs, trdata->ninputs);
  lar_create_matrix(&trdata->outputs, trdata->nobs, trdata->noutputs);

  i = j = 0;
  last = tmp = buffer;
  while (*tmp) {
    if (*tmp == ' ') {
      *tmp = '\0';
      if (j < trdata->ninputs) {
	*(trdata->inputs.v[i][j]) = atof(last);
      } else {
	*(trdata->outputs.v[i][j - trdata->ninputs]) = atof(last);
      }
      j++;
      tmp++;
      last = tmp;
    } else if (*tmp == '\n') {
      *tmp = '\0';
      if (j < trdata->ninputs) {
	*(trdata->inputs.v[i][j]) = atof(last);
      } else {
	*(trdata->outputs.v[i][j - trdata->ninputs]) = atof(last);
      }
      j = 0;
      i++;
      tmp++;
      last = tmp;
    } else {
      tmp++;
    }
  }
  free(buffer);
}


void nn_load_trdata(struct lar_matrix *trinput, struct lar_matrix *troutput, int ninputs, int noutputs, FILE *fp, char *delim)
{
  unsigned long nobs;
  int h, i, j;  
  long size;
  char *tmp;
  char *last;
  char *buffer;
  struct stat st;
  int buffer_size;
  
  fstat(fileno(fp), &st);
  size = st.st_size;
  buffer_size = 2048 * ((size / 2048) + 1);
  buffer = calloc(buffer_size, sizeof (char));
  fread(buffer, 1, size, fp);

  /*
  nobs = 0;
  tmp = buffer;
  while (*tmp) {
    if (*tmp == '\n') {
      nobs++;
    }
    tmp++;
  }
  */

  nobs = nn_nchar(buffer, "\n");
  
  lar_create_matrix(trinput, nobs, ninputs);
  lar_create_matrix(troutput, nobs, noutputs);

  i = j = 0;
  last = tmp = buffer;
  while (*tmp) {
    if (*tmp == ' ') {
      *tmp = '\0';
      if (j < ninputs) {
	*(trinput->v[i][j]) = atof(last);
      } else {
	*(troutput->v[i][j - ninputs]) = atof(last);
      }
      j++;
      tmp++;
      last = tmp;
    } else if (*tmp == '\n') {
      *tmp = '\0';
      if (j < ninputs) {
	*(trinput->v[i][j]) = atof(last);
      } else {
	*(troutput->v[i][j - ninputs]) = atof(last);
      }
      j = 0;
      i++;
      tmp++;
      last = tmp;
    } else {
      tmp++;
    }
  }
  free(buffer);
}
