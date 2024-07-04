#!/bin/bash
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
declare -A domains=( ["Hotel"]="vlsp2018_hotel" ["Restaurant"]="vlsp2018_restaurant" )
approaches=("v1" "v2")

for domain in "${!domains[@]}"; do
    test_dataset=$SCRIPT_DIR/../datasets/${domains[$domain]}/3-VLSP2018-SA-$domain-test.txt
    for approach in "${approaches[@]}"; do
        pred_file=$SCRIPT_DIR/../experiments/predictions/$domain-$approach.txt
        java $SCRIPT_DIR/SAEvaluate.java $test_dataset $pred_file > $SCRIPT_DIR/$domain-$approach-eval.txt
    done
done