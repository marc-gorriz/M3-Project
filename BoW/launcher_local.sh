#!/bin/bash

DATA_PATH="../../Databases/MIT_split"
MODEL_PATH="../../Lab2-BoW/test1/model.pkl"
CODEBOOK_PATH="../../Lab2-BoW/test1/"
VISUALWORDS_PATH="../../Lab2-BoW/test1/"
EVALUATION_PATH="../../Lab2-BoW/test1/confusion_matrix.png"

if [ "$1" = "train" ]
then
    python3 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --train --compute_features
elif [ "$1" = "test" ]
then
    python3 main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --test --compute_features
else
    echo "Incorrect option"
fi