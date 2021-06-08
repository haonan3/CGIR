#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# First step, data cleaning.
# Please download ml-20m and ml-25m from https://grouplens.org/datasets/movielens/,
# and then put them to data/ml-20m/raw/ and data/ml-20m/raw/ respectively.
python src/preprocess_ml.py --dataset_name ml-20m
sleep 5
python src/preprocess_ml.py --dataset_name ml-25m


# Run exp
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name gradientIR --data ml-20m --modification_level 1 --epochs 200 \
                --word_batch_size 1024 --disentanglement_batch_size 1024 --cl_batch_size 1024 --eval_on_all 1  \
                --sampled_test 0 --norm_CL_vec 0 --norm_CL_score 1 --test_interval 10 --gradient_test 0 --vis 0 \
                --independence_test 1 --IR_test 1 --eval_batch 15 --include_zero_dist 0 --beta 0.5 &
sleep 5
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name gradientIR --data ml-25m --modification_level 1 --epochs 200 \
                --word_batch_size 1024 --disentanglement_batch_size 1024 --cl_batch_size 1024 --eval_on_all 1  \
                --sampled_test 0 --norm_CL_vec 0 --norm_CL_score 1 --test_interval 10 --gradient_test 0 --vis 0 \
                --independence_test 1 --IR_test 1 --eval_batch 15 --include_zero_dist 0 --beta 0.5 &
