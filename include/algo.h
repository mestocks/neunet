#ifndef algo_h__
#define algo_h__

extern double sigmoid(double x);
extern double dsigmoid(double x);

extern int randint(int n);

extern void feed_forward(struct Layer *head_layer, int nlayers, double *inputs, int ninputs);
extern void deltas(struct Layer *head_layer, int nlayers, double *inputs, double *targets);
extern void back_propagate(struct Layer *head_layer, int nlayers, double *inputs, double *targets, double lrate);

#endif
