#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorriz/work/master/models/session02/test2/model.pkl"
CODEBOOK_PATH="/imatge/mgorriz/work/master/models/session02/test2/"
VISUALWORDS_PATH="/imatge/mgorriz/work/master/models/session02/test1/"
EVALUATION_PATH="/imatge/mgorriz/work/master/evaluation/session02/test2/confusion_matrix.png"

DESCRIPTOR="bow_dense_sift"

if [ "$1" = "train" ]
then
    if [ "$2" = "compute" ]
    then
        srun --pty -c 5 --mem 30G python2 -i main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --descriptor $DESCRIPTOR --train --compute_features
    else
        srun --pty -c 5 --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --descriptor $DESCRIPTOR --train
    fi
elif [ "$1" = "test" ]
then
    if [ "$2" = "compute" ]
    then
        srun --pty -c 5 --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --descriptor $DESCRIPTOR --test --compute_features
    else
        srun --pty -c 5 --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --descriptor $DESCRIPTOR --test
    fi
else
    echo "Incorrect option"
fi