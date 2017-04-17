#include <stdlib.h>
#include <math.h>

#include <neurons.h>
#include <algo.h>

double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

double dsigmoid(double x)
{
  return x * (1 - x);
}

int get_weight_index(struct Layer *layer, int i, int h)
{
  // [i1h1, i2h1, i3h1, i1h2, ...]
  return (h * layer->ninputs) + i;
}


void feed_forward(struct Layer *head_layer, double *inputs, int ninputs)
{
  int h, i;
  int windex;
  double *tmp_inputs;
  struct Layer *curr_layer;

  tmp_inputs = inputs;
  curr_layer = head_layer;
  while (curr_layer != NULL) {
    for (h = 0; h < curr_layer->nhidden; h++) {
      curr_layer->tmp_h[h] = 0.0;
      for (i = 0; i < curr_layer->ninputs; i++) {
	windex = get_weight_index(curr_layer, i, h);
	curr_layer->tmp_h[h] += tmp_inputs[i] * curr_layer->input_weights[windex];

      }
      curr_layer->hidden[h] = sigmoid(curr_layer->tmp_h[h]);
    }
    tmp_inputs = &curr_layer->hidden[0];
    curr_layer = curr_layer->next;
  }
}

void back_propogate(struct Layer *tail_layer, double *inputs, double *targets, double lrate)
{
  int h, i;
  int windex;
  double tmp_dsig;
  double tmp_error;
  struct Layer *curr_layer;
  
  curr_layer = tail_layer;
  double zero_wdelta_sum;

  zero_wdelta_sum = 0.0;
  for (h = 0; h < curr_layer->nhidden; h++) {
    tmp_dsig = dsigmoid(curr_layer->hidden[h]);
    tmp_error = targets[h] - curr_layer->hidden[h];
    curr_layer->deltas[h] = tmp_dsig * tmp_error;
    for (i = 0; i < curr_layer->ninputs; i++) {
      windex = get_weight_index(curr_layer, i, h);
      curr_layer->input_weights[windex] += lrate * curr_layer->deltas[h] * curr_layer->prev->hidden[i];
      curr_layer->wdelta_sum[i] = (zero_wdelta_sum * curr_layer->wdelta_sum[i]) + (curr_layer->input_weights[windex] * curr_layer->deltas[h]);
    }
    zero_wdelta_sum = 1.0;
  }

  curr_layer = curr_layer->prev;

  while (curr_layer->prev != NULL) {
    zero_wdelta_sum = 0.0;
    for (h = 0; h < curr_layer->nhidden; h++) {
      tmp_dsig = dsigmoid(curr_layer->hidden[h]);
      curr_layer->deltas[h] = tmp_dsig * curr_layer->next->wdelta_sum[h];
      for (i = 0; i < curr_layer->ninputs; i++) {
	windex = get_weight_index(curr_layer, i, h);
	curr_layer->input_weights[windex] += lrate * curr_layer->deltas[h] * curr_layer->prev->hidden[i];
	curr_layer->wdelta_sum[i] = (zero_wdelta_sum * curr_layer->wdelta_sum[i]) + (curr_layer->input_weights[windex] * curr_layer->deltas[h]);
      }
      zero_wdelta_sum = 1.0;      
    }
    curr_layer = curr_layer->prev;
  }
    
  zero_wdelta_sum = 0.0;
  for (h = 0; h < curr_layer->nhidden; h++) {
    tmp_dsig = dsigmoid(curr_layer->hidden[h]);
    curr_layer->deltas[h] = tmp_dsig * curr_layer->next->wdelta_sum[h];
    for (i = 0; i < curr_layer->ninputs; i++) {
      windex = get_weight_index(curr_layer, i, h);
      curr_layer->input_weights[windex] += lrate * curr_layer->deltas[h] * inputs[i];
      curr_layer->wdelta_sum[i]  = (zero_wdelta_sum * curr_layer->wdelta_sum[i]) + (curr_layer->input_weights[windex] * curr_layer->deltas[h]);
    }
    zero_wdelta_sum = 1.0;
  }
}
