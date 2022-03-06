#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-p100-002
#$-l gpu_card=4
#$-N CRC_gen_model_pred


export PATH=/nfs/yding4/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/gen_model_pred.py

INPUT_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att
TEST_DIR=/nfs/yding4/AVE_project/consumable/clean_test_data

# 15att
#ATT_LIST="['ActiveIngredients','AgeRangeDescription','Color','FinishType','Flavor','HairType','ItemForm','Material','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','Variety']"
ATT_LIST="['ActiveIngredients']"

/nfs/yding4/conda_envs/W2NER/bin/python ${CODE} \
    --input_dir ${INPUT_DIR}    \
    --test_dir ${TEST_DIR}    \
    --att_list ${ATT_LIST}