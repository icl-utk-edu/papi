#!/bin/bash

if [ $# -lt 1 ]
then
    printf "replicate.sh: Must supply output dir for generated source!\n"
    exit 1
fi
for ((i=1; i<12; i++)); do
    cp $1/icache_seq_kernel_0.so $1/icache_seq_kernel_${i}.so
done
