#ifndef algo_h__
#define algo_h__

extern double sigmoid(double x);
extern double dsigmoid(double x);
extern int get_weight_index(struct Layer *layer, int i, int h);
extern void feed_forward(struct Layer *head_layer, int nlayers, double *inputs, int ninputs);
extern void back_propagate(struct Layer *head_layer, int nlayers, double *inputs, double *targets, double lrate);

#endif
