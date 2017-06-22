HOME = $(shell echo $$HOME)/
BASE = $(HOME)repos/
PREFIX = $(HOME).local/

###

bin = bin/
inc = include/
misc = misc/
src = src/

Wgcc = -Wall -Wextra -Wpedantic

###

lib = lar rawk
slib = lar m rawk

ext-inc = $(addprefix $(PREFIX)include/lib,$(lib))
all-inc = $(inc) $(ext-inc)
ext-lib = $(PREFIX)lib/
all-lib = $(ext-lib)

INC = $(addprefix -I ,$(all-inc))
LIB = $(addprefix -L ,$(all-lib))
SLIB = $(addprefix -l,$(slib))


###

.PHONY:	all
all:	$(bin)synapse

$(bin)%:	$(src)main.c $(src)nn_algo.c $(src)nn_objects.c
	gcc $(INC) $(LIB) $(Wgcc) -o $@ $^ $(SLIB)


###

.PHONY:	test
test:	$(misc)profile

$(misc)%:	$(misc)%.c $(src)nn_algo.c $(src)nn_objects.c
	gcc $(INC) $(LIB) $(Wgcc) -o $@ $^ $(SLIB)

















.PHONY:	old
old:	$(bin)perceptron

$(bin)perceptron:	$(src)perceptron.c $(src)algo.c $(src)neurons.c $(inc)algo.h $(inc)neurons.h
	gcc -I $(inc) -I ~/.local/include/librawk/ -L ~/.local/lib/ -o $@ $(word 1,$^) $(word 2,$^) $(word 3,$^) -lm -lrawk

