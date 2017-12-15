#!/bin/bash

DATA_PATH="data_path"
MODEL_PATH="model_path"
CLASSIFIER="knn"
TRAIN_METHOD="kfold"
K_FOLD_K=5


if [ "$1" = "train" ]
then
    python3 main.py --data_path $DATA_PATH --output_path MODEL_PATH --classifier CLASSIFIER --train_method TRAIN_METHOD --kfold_k K_FOLD_K --train
elif [ "$1" = "test" ]
then
    python3 main.py --data_path $DATA_PATH --output_path MODEL_PATH --classifier CLASSIFIER --train_method TRAIN_METHOD --kfold_k K_FOLD_K --test
else
    echo "Incorrect option"
fi