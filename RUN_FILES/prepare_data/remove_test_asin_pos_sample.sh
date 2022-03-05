#!/bin/bash

export PATH=/nfs/yding4/conda_envs/AVEQA_PyTorch2/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/AVEQA_PyTorch2/lib:$LD_LIBRARY_PATH

INPUT_FILE=/nfs/yding4/AVE_project/consumable/title_bullet/33att/sample_data_1/pos_sample.jsonl
INPUT_GT_DIR=/nfs/yding4/AVE_project/consumable/clean_test_data
OUTPUT_DIR=/nfs/yding4/AVE_project/consumable/title_bullet/33att/sample_data_1/
OUTPUT_FILE=pos_sample
RATIO=0.8

CODE=/nfs/yding4/AVE_project/AVEQA_PyTorch/AVEQA_PyTorch/preprocessing/prepare_data/remove_test_asin_split_train_valid.py

 /nfs/yding4/conda_envs/AVEQA_PyTorch2/bin/python ${CODE} \
  --input_file  ${INPUT_FILE} \
  --input_gt_dir  ${INPUT_GT_DIR} \
  --ratio ${RATIO}  \
  --output_file ${OUTPUT_FILE}  \
  --output_dir ${OUTPUT_DIR}