#! /bin/bash

# usage:
#
#     synapse-run.sh <arch> <nepochs>

BASE=$HOME/repos/neunet
BIN=$BASE/bin
MISC=$BASE/misc
MNIST=$BASE/mnist

arch=$1
nepochs=$2

python $MISC/net_arch.py $arch > $MNIST/weights0.txt

for ((i=1; i<=$nepochs; i++)); do
    j=$((i-1))
    $BIN/synapse learn $arch 10 10 $MNIST/weights$j.txt $MNIST/mnist_train_wocvs_edited_norm.txt > $MNIST/weights$i.txt

    $BIN/synapse solve $arch 10 10 $MNIST/weights$i.txt < $MNIST/mnist_cvs_edited_norm.txt | awk ' { maxindex=0; max=0; for (i=1; i<=NF; i++) { if ($i > max) { max=$i; maxindex=i } } print maxindex-1 } ' > $MNIST/mnist_cvs_edited_norm.slv$i

    paste -d ' ' $MNIST/mnist_cvs_edited_norm.ans $MNIST/mnist_cvs_edited_norm.slv$i | awk ' { if ($1==$2) a++; n++ }; END { print a,n,a/n } '
done
