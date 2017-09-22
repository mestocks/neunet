#ifndef nn_matrix_h__
#define nn_matrix_h__
#endif

#ifndef nn_objects_h__
#define nn_objects_h__
#endif

#ifndef nn_algo_h__
#define nn_algo_h__


static inline double nn_sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

static inline double nn_dsigmoid(double x)
{
  return x * (1-x);
}


static inline double nn_tanh(double x)
{
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

static inline double nn_dtanh(double x)
{
  return 1 - (x * x);
}


static inline double nn_ReLU(double x)
{
    return x > 0 ? x : 0;
}

static inline double nn_dReLU(double x)
{
    return x < 0 ? 0 : 1;
}


static inline double nn_lReLU(double x)
{
  return x > 0.01 * x ? x : 0.01 * x;
}

static inline double nn_dlReLU(double x)
{
  return x < 0 ? 0.01 : 1;
}

extern void minibatch_feed_forward(struct NeuNet *nnet);
extern void minibatch_back_propagation(struct NeuNet *nnet);
extern void minibatch_update_weights(struct NeuNet *nnet, unsigned long m, double lambda, int reg, double lrate);

#endif
