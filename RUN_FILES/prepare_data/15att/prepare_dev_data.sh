#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-001
#$-l gpu_card=4
#$-N CRC_ran_end2end

export PATH=/nfs/yding4/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/prepare_train_data.py

INPUT_INSTANCE_FILE=/nfs/yding4/AVE_project/consumable/title_bullet/15att/sample_data_1/pos_sample.dev
OUTPUT_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/15att
OUTPUT_FILE=dev.json
MAX_SPAN_LENGTH=5
MAX_SEQ_LENGTH=100

# 15att
ATT_LIST="['ActiveIngredients','AgeRangeDescription','Color','FinishType','Flavor','HairType','ItemForm','Material','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','Variety']"

/nfs/yding4/conda_envs/W2NER/bin/python ${CODE}   \
    --input_instance_file ${INPUT_INSTANCE_FILE}  \
    --output_dir ${OUTPUT_DIR}  \
    --output_file ${OUTPUT_FILE}    \
    --max_span_length ${MAX_SPAN_LENGTH}    \
    --max_seq_length    ${MAX_SEQ_LENGTH}   \
    --att_list ${ATT_LIST}