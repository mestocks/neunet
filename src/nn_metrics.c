#include <math.h>
#include <stdlib.h>

#include <lar_objects.h>
#include <lar_init.h>
#include <nn_objects.h>
#include <nn_algo.h>

double nn_error(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int m, j;
  double error;
  double tmp_error;
  double exp;
  double obs;
  struct lar_matrix *a;

  error = 0.0;
  a = nnet->layers[nnet->nlayers - 1];
  for (m = 0; m < trdata->nobs; m++) {
    for (j = 0; j < trdata->ninputs; j++) {
      *(nnet->layers[0]->v[j][0]) = *(trdata->inputs.v[m][j]);
    }
    for (j = 0; j < trdata->noutputs; j++) {
      *y->v[j][0] = *(trdata->outputs.v[m][j]);
    }
    nn_feed_forward(nnet);

    for (j = 0; j < a->nrows; j++) {
      exp = *y->v[j][0];
      obs = *a->v[j][0];
      tmp_error = obs - exp;
      error += tmp_error * tmp_error;
    }
  }
  return error / (2 * trdata->nobs);
}

double nn_cost(struct TrainingData *trdata, struct NeuralNetwork *nnet, struct lar_matrix *y)
{
  int m, j;
  double cost;
  double tmp_cost;
  double exp;
  double obs;
  struct lar_matrix *a;

  cost = 0.0;
  a = nnet->layers[nnet->nlayers - 1];
  for (m = 0; m < trdata->nobs; m++) {
    for (j = 0; j < trdata->ninputs; j++) {
      *(nnet->layers[0]->v[j][0]) = *(trdata->inputs.v[m][j]);
    }
    for (j = 0; j < trdata->noutputs; j++) {
      *y->v[j][0] = *(trdata->outputs.v[m][j]);
    }
    nn_feed_forward(nnet);

    for (j = 0; j < a->nrows; j++) {
      exp = *y->v[j][0];
      obs = *a->v[j][0];
      tmp_cost = (-exp * log(obs)) - ((1 - exp) * log(1 - obs));
      cost += tmp_cost;
    }
  }
  return cost / trdata->nobs;
}
