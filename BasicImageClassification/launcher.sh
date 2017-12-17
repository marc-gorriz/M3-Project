#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorriz/work/master/models/session01/knn_surf/kfold5_model_4.pkl"
EVALUATION_PATH="/imatge/mgorriz/work/master/evaluation/session01/knn_surf/confusion_matrix_kfold5_4.png"
FEATURES_DESCRIPTOR='surf'
CLASSIFIER="knn"
TRAIN_METHOD="kfold"

if [ "$1" = "train" ]
then
    srun --pty -c 8 --mem 30G python2 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --evaluation_path $EVALUATION_PATH --descriptor $FEATURES_DESCRIPTOR --classifier $CLASSIFIER --train_method $TRAIN_METHOD --train
elif [ "$1" = "test" ]
then
    srun --pty -c 2 --mem 30G python2 main.py  --data_path $DATA_PATH --model_path $MODEL_PATH --evaluation_path $EVALUATION_PATH --descriptor $FEATURES_DESCRIPTOR --classifier $CLASSIFIER --train_method $TRAIN_METHOD --test
else
    echo "Incorrect option"
fi