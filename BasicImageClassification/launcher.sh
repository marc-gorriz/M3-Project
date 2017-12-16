#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorriz/work/master/models/session01/kfold_model_100.pkl"
EVALUATION_PATH="/imatge/mgorriz/work/master/evaluation/session01/confusion_matrix_kfold_100.png"
CLASSIFIER="knn"
TRAIN_METHOD="kfold"

if [ "$1" = "train" ]
then
    srun --pty --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --evaluation_path $EVALUATION_PATH --classifier $CLASSIFIER --train_method $TRAIN_METHOD --train
elif [ "$1" = "test" ]
then
    srun --pty --mem 30G python2 main.py  --data_path $DATA_PATH --model_path $MODEL_PATH --evaluation_path $EVALUATION_PATH --classifier $CLASSIFIER --train_method $TRAIN_METHOD --test
else
    echo "Incorrect option"
fi
