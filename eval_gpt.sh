#!/bin/bash

base_dir="zeroshot/llama3_3b"

# Associate each integer value with a list of corresponding dataset names
declare -A dataset_map
# List of datasets that are not multi-choice
dataset_map[0]="Fed-aya-0 Fed-aya-1 Fed-aya-2"
dataset_map[1]="Fed-aya-10 Fed-aya-11 Fed-aya-12"
dataset_map[2]="Fed-aya-20 Fed-aya-21 Fed-aya-22"
dataset_map[3]="Fed-aya-30 Fed-aya-31 Fed-aya-32"
dataset_map[4]="Fed-aya-40 Fed-aya-41 Fed-aya-42"
dataset_map[5]="Fed-aya-50 Fed-aya-51 Fed-aya-52"
dataset_map[6]="Fed-aya-60 Fed-aya-61 Fed-aya-62"
dataset_map[7]="Fed-aya-70 Fed-aya-71 Fed-aya-72"
dataset_map[8]="Fed-aya-80 Fed-aya-81 Fed-aya-82"
dataset_map[9]="Fed-aya-90 Fed-aya-91 Fed-aya-92"

round=15

# Iterate over the keys (integer values) in the associative array
for i in "${!dataset_map[@]}"; do
    datasets=(${dataset_map[$i]})
    for dataset in "${datasets[@]}"; do
        input_file="./eval_results/${base_dir}/client${i}_round${round}_${dataset}.json"
        output_file="./eval_results_gpt/${base_dir}/client${i}_round${round}_${dataset}.jsonl"
        
        echo "Processing ${dataset} (client ${i})..."
        OPENAI_API_KEY="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" \
        python eval_gpt_aya.py -r "$input_file" -o "$output_file" --random_seed 42
    done
done

echo "All datasets processed."
