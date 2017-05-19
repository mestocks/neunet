#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <neurons.h>
#include <algo.h>

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dsigmoid(x) (x * (1-x))

/*
double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}
*/

double dsigmoid(double x)
{
  return x * (1 - x);
}


int randint(int max)
{
  int i, r;
  long end;

  if ((max - 1) == RAND_MAX) {
    return rand();
  } else {
    end = RAND_MAX / max;
    assert (end > 0L);
    end *= max;
    while ((r = rand()) >= end);
    return r % max;
  }
}


void feed_forward(struct Layer *head_layer, int nlayers, double *inputs, int ninputs)
{
  int l, h, i;
  double *tmp_inputs;

  tmp_inputs = inputs;
  for (l = 0; l < nlayers; l++) {
    for (h = 0; h < head_layer[l].nhidden; h++) {
      head_layer[l].tmp_h[h] = 0.0;
      for (i = 0; i < head_layer[l].ninputs; i++) {
	head_layer[l].tmp_h[h] += tmp_inputs[i] * head_layer[l].weights[i][h];
      }
      head_layer[l].hidden[h] = sigmoid(head_layer[l].tmp_h[h]);
    }
    tmp_inputs = &head_layer[l].hidden[0];
  }
}

void loss_function(double *error, int nvalues, double *targets, double *outputs)
{
  int i;
  double diff;
  
  for (i = 0; i < nvalues; i++) {
    error[i] = targets[i] - outputs[i];
  }
}

void deltas(struct Layer *head_layer, int nlayers, double *inputs, double *targets)
{
  int l, h, i;
  double tmp_error;
  
  for (l = nlayers - 1; l >= 0; l--) {
    for (h = 0; h < head_layer[l].nhidden; h++) {
      if (l == nlayers - 1) {
	tmp_error = targets[h] - head_layer[l].hidden[h];
	head_layer[l].deltas[h] += dsigmoid(head_layer[l].hidden[h]) * tmp_error;
      } else {
	head_layer[l].deltas[h] += dsigmoid(head_layer[l].hidden[h]) * head_layer[l + 1].wdelta_sum[h];
      }
      for (i = 0; i < head_layer[l].ninputs; i++) {
	if (h == 0) {
	  head_layer[l].wdelta_sum[i] = head_layer[l].weights[i][h] * head_layer[l].deltas[h];
	} else {
	  head_layer[l].wdelta_sum[i] += head_layer[l].weights[i][h] * head_layer[l].deltas[h];
	}
      }
    }
  }
}

void back_propagate(struct Layer *head_layer, int nlayers, double *inputs, double *targets, double lrate)
{
  int l, h, i;
  double delta_lrate;
  double tmp_error;
  
  for (l = nlayers - 1; l >= 0; l--) {
    for (h = 0; h < head_layer[l].nhidden; h++) {
      
      if (l == nlayers - 1) {
	tmp_error = targets[h] - head_layer[l].hidden[h];
	head_layer[l].deltas[h] = dsigmoid(head_layer[l].hidden[h]) * tmp_error;
      } else {
	head_layer[l].deltas[h] = dsigmoid(head_layer[l].hidden[h]) * head_layer[l + 1].wdelta_sum[h];
      }
      delta_lrate = lrate * head_layer[l].deltas[h];
      
      for (i = 0; i < head_layer[l].ninputs; i++) {
	if (l == 0) {
	  head_layer[l].weights[i][h] += delta_lrate * inputs[i];
	} else {
	  head_layer[l].weights[i][h] += delta_lrate * head_layer[l - 1].hidden[i];
	}
		
	if (h == 0) {
	  head_layer[l].wdelta_sum[i] = head_layer[l].weights[i][h] * head_layer[l].deltas[h];
	} else {
	  head_layer[l].wdelta_sum[i] = head_layer[l].wdelta_sum[i] + (head_layer[l].weights[i][h] * head_layer[l].deltas[h]);
	}
      }
    }
  }
}
