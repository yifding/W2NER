#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-p100-001
#$-l gpu_card=1
#$-N CRC_W2NER_48AE

export PATH=/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE_DIR=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER

CONFIG=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER/RUN_FILES/train_15att/15att.config
DATASET_DIR=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/48AE
MODEL_DIR=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER/RUN_FILES/train_48AE

# 48att = 50att - ReleaseDate - ShoeWidth
ATT_LIST="['ApplicablePlace','AthleticShoeType','BackSideMaterial','BodyMaterial','BrandName','Capacity','Category','ClosureType','Collar','Color','DepartmentName','DerivativeSeries','FabricType','Feature','FingerboardMaterial','Fit','Function','Gender','HoseHeight','InsoleMaterial','IsCustomized','ItemType','Length','LensesOpticalAttribute','LevelOfPractice','LiningMaterial','Material','Model','ModelNumber','Name','OuterwearType','OutsoleMaterial','PatternType','ProductType','Season','Size','SleeveLengthCm','SportType','SportsType','StrapType','Style','Technology','Type','TypeOfSports','UpperHeight','UpperMaterial','Voltage','Weight']"


cd ${CODE_DIR}

/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/bin/python main.py    \
    --device    3   \
    --config    ${CONFIG}   \
    --dataset_dir   ${DATASET_DIR}  \
    --model_dir ${MODEL_DIR}    \
    --att_list  ${ATT_LIST}
