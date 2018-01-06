#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorriz/work/master/models/session02/test1/model.pkl"
CODEBOOK_PATH="/imatge/mgorriz/work/master/models/session02/test1/"
VISUALWORDS_PATH="/imatge/mgorriz/work/master/models/session02/test1/"
EVALUATION_PATH="/imatge/mgorriz/work/master/evaluation/session02/test1/confusion_matrix_hog.png"

if [ "$1" = "train" ]
then
    srun --pty -c 5 --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --train --compute_features
elif [ "$1" = "test" ]
then
    srun --pty -c 5 --mem 30G python2 -i main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --test --compute_features
else
    echo "Incorrect option"
fi