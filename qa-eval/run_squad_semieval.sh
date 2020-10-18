#!/bin/bash

set -e
set -x

batch_size=32
devices="0_1_2_3"
pretrain_file="../data/harv_synthetic_data_semi/0.4_replaced_1.0_harv_features.txt"
list="0.1"
dist_url="tcp://127.0.0.1:9999"

for i in $list
do
    python main.py \
    --batch_size $batch_size \
    --devices $devices \
    --pretrain_file $pretrain_file \
    --unlabel_ratio $i \
    --dist_url $dist_url \
    --lazy_loader
done
