#!/bin/bash
python generate_image_256_512.py \
    --network=./network-snapshot-000860.pkl \
    --dpath=./output_path/one_result \
    --resolution=512 \
    --outdir=./output_path/512_result \

python generate_image_512_1024.py \
    --network=./network-snapshot-000860.pkl \
    --dpath=./output_path/512_result \
    --resolution=1024 \
    --outdir=./output_path/1024_result \

