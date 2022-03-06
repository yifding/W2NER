#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-001
#$-l gpu_card=4
#$-N CRC_ran_end2end

export PATH=/nfs/yding4/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE_DIR=/nfs/yding4/AVE_project/W2NER

CONFIG=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att/33att_config.json

PRETRAINED_CHECKPOINT=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att/epoch-9.pt
OUTPUT_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att/raw_prediction
TEST_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/test
ATT_LIST="['ActiveIngredients','AgeRangeDescription','BatteryCellComposition','Brand','CaffeineContent','CapacityUnit','CoffeeRoastType','Color','DietType','DosageForm','EnergyUnit','FinishType','Flavor','FormulationType','HairType','Ingredients','ItemForm','ItemShape','LiquidContentsDescription','Material','MaterialFeature','MaterialTypeFree','PackageSizeName','Pattern','PatternType','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','TeaVariety','Variety']"
#ATT_LIST="['ActiveIngredients','AgeRangeDescription','BatteryCellComposition','Brand','CaffeineContent','CapacityUnit','CoffeeRoastType','Color','DietType','DosageForm','EnergyUnit','FinishType','Flavor','FormulationType','HairType','Ingredients','ItemForm','ItemShape','LiquidContentsDescription','Material','MaterialFeature','MaterialTypeFree','PackageSizeName','Pattern','PatternType','ProductBenefit','Scent']"
TEST_ATT_LIST="['ActiveIngredients','AgeRangeDescription','Color','FinishType','Flavor','HairType','ItemForm','Material','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','Variety']"

cd ${CODE_DIR}

/nfs/yding4/conda_envs/W2NER/bin/python yd_evaluation.py    \
    --device    1   \
    --config    ${CONFIG}   \
    --pretrained_checkpoint ${PRETRAINED_CHECKPOINT}  \
    --output_dir ${OUTPUT_DIR}    \
    --test_dir ${TEST_DIR}    \
    --att_list ${ATT_LIST}    \
    --test_att_list ${TEST_ATT_LIST}