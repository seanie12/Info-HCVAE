#!/bin/bash

set -e
set -x

gpu=0,1,3
batch_size=24
devices="0_1_2"
workers=3

pretrain_file = "../data/harv_synthetic_data_qae/1.0_harv_features.txt"
data_name="squad"
list="1.0"
dist_url="tcp://127.0.0.1:9991"

export CUDA_VISIBLE_DEVICES=$gpu

for i in $pretrain_file
do
    python main.py \
    --batch_size $batch_size \
    --devices $devices \
    --workers $workers \
    --pretrain_file $i \
    --data_name $data_name \
    --lazy_loader \
    --unlabel_ratio 1.0 \
    --dist_url $dist_url
done
