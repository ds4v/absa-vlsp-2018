#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

declare -A domains=( ["Hotel"]="vlsp2018_hotel" ["Restaurant"]="vlsp2018_restaurant" )
approaches=("v1" "v2")

for domain in "${!domains[@]}"; do
    dataset="$PROJECT_DIR/datasets/${domains[$domain]}/3-VLSP2018-SA-${domain}-test.txt"
    for approach in "${approaches[@]}"; do
        prediction="$PROJECT_DIR/experiments/predictions/${domain}-${approach}.txt"
        output="$SCRIPT_DIR/${domain}-${approach}-eval.txt"
        java "$SCRIPT_DIR"/SAEvaluate.java "$dataset" "$prediction" > "$output"
    done
done