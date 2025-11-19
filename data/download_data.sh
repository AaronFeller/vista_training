#!/bin/sh

mkdir -p train
cd train

for i in $(seq -w 000 049); do
    wget https://huggingface.co/datasets/nvidia/esm2_uniref_pretraining_data/resolve/main/train/${i}.parquet
done

# cd ..
# mkdir -p test
# cd test

# wget https://huggingface.co/datasets/nvidia/esm2_uniref_pretraining_data/resolve/main/validation/000.parquet