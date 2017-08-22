AUTOMAKE_OPTIONS = foreign

ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = bin/neunet

bin_neunet_SOURCES = src/main.c src/nn_algo.c src/nn_objects.c src/nn_fileIO.c src/nn_metrics.c src/nn_args.c include/nn_algo.h include/nn_objects.h include/nn_fileIO.h include/nn_metrics.h include/nn_args.h
bin_neunet_CFLAGS = -I $(srcdir)/include -I $(includedir)/liblar -I $(includedir)/librawk

