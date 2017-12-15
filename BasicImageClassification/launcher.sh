#!/bin/bash

DATA_PATH="/imatge/mgorriz/work/master/Databases/MIT_split"
MODEL_PATH="/imatge/mgorrizwork/master/models/session01"
CLASSIFIER="knn"
TRAIN_METHOD="fixed"
K_FOLD_K=5


if [ "$1" = "train" ]
then
    srun --pty --mem 30G python3 main.py --data_path $DATA_PATH --output_path MODEL_PATH --classifier CLASSIFIER --train_method TRAIN_METHOD --kfold_k K_FOLD_K --train
elif [ "$1" = "test" ]
then
    srun --pty --mem 30G python3 main.py --data_path $DATA_PATH --output_path MODEL_PATH --classifier CLASSIFIER --train_method TRAIN_METHOD --kfold_k K_FOLD_K --test
else
    echo "Incorrect option"
fi
