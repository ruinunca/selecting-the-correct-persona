#!/bin/bash

FILES=$1/*

for f in $FILES
do
    if [[ "$f" != *"scores_"* ]]; then 
        echo "Processing $f file..."
        python evaluate_results.py --path $f
    fi
done