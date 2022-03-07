#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-001
#$-l gpu_card=4
#$-N CRC_ran_end2end

export PATH=/nfs/yding4/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE_DIR=/nfs/yding4/AVE_project/W2NER

CONFIG=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att/33att.config
DATASET_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/33att
MODEL_DIR=/nfs/yding4/AVE_project/W2NER/RUN_FILES/train_33att
ATT_LIST="['ActiveIngredients','AgeRangeDescription','BatteryCellComposition','Brand','CaffeineContent','CapacityUnit','CoffeeRoastType','Color','DietType','DosageForm','EnergyUnit','FinishType','Flavor','FormulationType','HairType','Ingredients','ItemForm','ItemShape','LiquidContentsDescription','Material','MaterialFeature','MaterialTypeFree','PackageSizeName','Pattern','PatternType','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','TeaVariety','Variety']"


cd ${CODE_DIR}

/nfs/yding4/conda_envs/W2NER/bin/python main.py    \
    --device    0   \
    --config    ${CONFIG}   \
    --dataset_dir   ${DATASET_DIR}  \
    --model_dir ${MODEL_DIR}    \
    --att_list  ${ATT_LIST}