#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <nn_matrix.h>
#include <nn_objects.h>
#include <nn_string.h>

#define MAX_WTS_RSIZE 5120

void nn_wts_from_file(struct NeuNet *nnet, char *fname)
{
  unsigned long b, i, j, l;
  FILE *fp;
  char **array;
  char delim = ' ';
  char buffer[MAX_WTS_RSIZE];

  fp = fopen(fname, "r");
  
  b = i = j = l = 0;
  
  array = calloc(1, sizeof (char*));
  while (fgets(buffer, sizeof(buffer), fp)) {
    nn_str2array(array, buffer, 1, &delim);
    if (b < nnet->bias_wts[l].ncols) {
      nnet->bias_wts[l].data[0][b] = atof(array[0]);
      b++;
    } else {
      nnet->weights[l].data[i][j] = atof(array[0]);
      j++;
    }
    if (j == nnet->weights[l].ncols) {
      j = 0;
      i++;
      if (i == nnet->weights[l].nrows) {
	i = 0;
	b = 0;
	l++;
      }
    }
  }
  free(array);
  fclose(fp);  
}


void nn_file2array(struct InOutData *iodata, FILE *fp, int ninputs, int noutputs, char *delim)
{
  unsigned long h, i, j, index, outdex;
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
  n = nn_nchar(buffer, "\n");

  iodata->input_data = calloc(n * ninputs, sizeof *iodata->input_data);
  iodata->output_data = calloc(n * noutputs, sizeof *iodata->output_data);

  create_smatrix(&iodata->inputs, n, ninputs);
  create_smatrix(&iodata->outputs, n, noutputs);

  i = j = index = outdex = 0;
  last = tmp = buffer;
  while (*tmp) {
    if (*tmp == ' ') {
      *tmp = '\0';
      if (j < ninputs) {
	iodata->input_data[index] = atof(last);
	index++;
      } else {
	iodata->output_data[outdex] = atof(last);
	outdex++;
      }
      j++;
      tmp++;
      last = tmp;
    } else if (*tmp == '\n') {
      *tmp = '\0';
      if (j < ninputs) {
	iodata->input_data[index] = atof(last);
	index++;
      } else {
	iodata->output_data[outdex] = atof(last);
	outdex++;
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

  attach_smatrix(&iodata->inputs, iodata->input_data);
  attach_smatrix(&iodata->outputs, iodata->output_data);
}
