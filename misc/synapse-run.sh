#! /bin/bash

# usage:
#
#     synapse-run.sh <arch> <nepochs> <training> <testing> <answers>
#
# 

BASE=$HOME/repos/neunet
BIN=$BASE/bin
MISC=$BASE/misc
MNIST=$BASE/mnist

arch=$1
nepochs=$2
TRAIN=$3
TEST=$4
ANS=$5

python $MISC/net_arch.py $arch > weights0.txt

for ((i=1; i<=$nepochs; i++)); do
    j=$((i-1))
    $BIN/synapse learn $arch 10 10 weights$j.txt $TRAIN > weights$i.txt

    #$BIN/synapse solve $arch 10 10 weights$i.txt < $TEST | awk ' { maxindex=0; max=0; for (i=1; i<=NF; i++) { if ($i > max) { max=$i; maxindex=i } } print maxindex-1 } ' > $TEST.slv$i
    $BIN/synapse solve $arch 10 10 weights$i.txt < $TEST | awk ' { if ($1 >=0.5) print 1; else print 0 } ' > $TEST.slv$i
    #$BIN/synapse solve $arch 10 10 weights$i.txt < $TEST

    paste -d ' ' $ANS $TEST.slv$i | awk ' { if ($1==$2) a++; n++ }; END { print a,n,a/n } '
done
