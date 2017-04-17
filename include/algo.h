#ifndef algo_h__
#define algo_h__

extern double sigmoid(double x);
extern double dsigmoid(double x);
extern int get_weight_index(struct Layer *layer, int i, int h);
extern void feed_forward(struct Layer *head_layer, double *inputs, int ninputs);
extern void back_propogate(struct Layer *tail_layer, double *inputs, double *targets, double lrate);

#endif
