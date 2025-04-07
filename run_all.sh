#!/bin/bash

STEP=200  # ステップサイズ

for START in 0 200 400 600 800
do
    echo "Running with start_index=$START"
    python generate_rome_2.py --start_index $START --step_size $STEP
done