#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#

algorithm=(knn mc_perceptron)
options=(data1)
path=(../dataset/Clayton_Kershaw/)
save_path=(Model/)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${save_path}${opt}.${algo}.model --data ${path}${opt}.train --online-training-iterations 10
        python classify.py --mode test --model-file ${save_path}${opt}.${algo}.model --data ${path}${opt}.test --predictions-file ${save_path}${opt}.test.predictions
        acc="$(python compute_accuracy.py ${path}${opt}.test ${save_path}${opt}.test.predictions)"
        echo "${opt} | $algo | $acc"
    done
done

$SHELL