AUTOMAKE_OPTIONS = foreign

ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = bin/synapse

bin_synapse_SOURCES = src/main.c src/nn_algo.c src/nn_objects.c src/nn_fileIO.c src/nn_metrics.c include/nn_algo.h include/nn_objects.h include/nn_fileIO.h include/nn_metrics.h
bin_synapse_CFLAGS = -I $(srcdir)/include -I $(includedir)/liblar -I $(includedir)/librawk

