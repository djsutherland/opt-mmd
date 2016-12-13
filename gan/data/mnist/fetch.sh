#!/usr/bin/env bash

cd $(dirname $0)

for fn in {train,t10k}-{images-idx3,labels-idx1}-ubyte; do
    if [[ ! -e $fn ]]; then
        wget -c http://yann.lecun.com/exdb/mnist/$fn.gz
        gunzip $fn.gz
    fi
done
