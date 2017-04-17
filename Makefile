bin = bin/
inc = include/
src = src/

.PHONY:	all
all:	$(bin)perceptron

$(bin)perceptron:	$(src)perceptron.c $(src)algo.c $(src)neurons.c $(inc)algo.h $(inc)neurons.h
	gcc -I $(inc) -I ~/.local/include/librawk/ -L ~/.local/lib/ -o $@ $(word 1,$^) $(word 2,$^) $(word 3,$^) -lm -lrawk
