#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorrizwork/master/models/session01"
CLASSIFIER="knn"
TRAIN_METHOD="fixed"

if [ "$1" = "train" ]
then
    srun --pty --mem 30G python2 -i main.py --data_path $DATA_PATH --model_path $MODEL_PATH --classifier $CLASSIFIER --train_method $TRAIN_METHOD --train
elif [ "$1" = "test" ]
then
    srun --pty --mem 30G python2 -i main.py --data_path $DATA_PATH --model_path $MODEL_PATH --classifier $CLASSIFIER --train_method $TRAIN_METHOD --test
else
    echo "Incorrect option"
fi
