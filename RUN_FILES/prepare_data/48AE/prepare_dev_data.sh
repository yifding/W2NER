#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-001
#$-l gpu_card=4
#$-N CRC_ran_end2end

export PATH=/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/lib:$LD_LIBRARY_PATH

CODE=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/48AE/prepare_train_data_48AE.py

INPUT_DIR=/afs/crc.nd.edu/user/y/yding4/AVE_project/AE/50att/split_by_att/train
OUTPUT_DIR=/afs/crc.nd.edu/user/y/yding4/AVE_project/W2NER/RUN_FILES/prepare_data/48AE
OUTPUT_FILE=dev.json
MAX_SPAN_LENGTH=5
MAX_SEQ_LENGTH=100
MODE="dev"

# 48att = 50att - ReleaseDate - ShoeWidth
ATT_LIST="['ApplicablePlace','AthleticShoeType','BackSideMaterial','BodyMaterial','BrandName','Capacity','Category','ClosureType','Collar','Color','DepartmentName','DerivativeSeries','FabricType','Feature','FingerboardMaterial','Fit','Function','Gender','HoseHeight','InsoleMaterial','IsCustomized','ItemType','Length','LensesOpticalAttribute','LevelOfPractice','LiningMaterial','Material','Model','ModelNumber','Name','OuterwearType','OutsoleMaterial','PatternType','ProductType','Season','Size','SleeveLengthCm','SportType','SportsType','StrapType','Style','Technology','Type','TypeOfSports','UpperHeight','UpperMaterial','Voltage','Weight']"


/afs/crc.nd.edu/user/y/yding4/AVE_project/conda_envs/W2NER/bin/python ${CODE}   \
    --input_dir ${INPUT_DIR}  \
    --output_dir ${OUTPUT_DIR}  \
    --output_file ${OUTPUT_FILE}    \
    --max_span_length ${MAX_SPAN_LENGTH}    \
    --max_seq_length    ${MAX_SEQ_LENGTH}   \
    --att_list ${ATT_LIST}  \
    --mode ${MODE}