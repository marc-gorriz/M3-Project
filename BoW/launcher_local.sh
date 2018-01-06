#!/bin/bash

DATA_PATH="../../Databases/MIT_split"
MODEL_PATH="../../Lab2-BoW/test1/hog_model.pkl"
CODEBOOK_PATH="../../Lab2-BoW/test1/codebook.pkl"
VISUALWORDS_PATH="../../Lab2-BoW/test1/visualwords.npy"
EVALUATION_PATH="../../Lab2-BoW/test1/confusion_matrix_hog.png"

if [ "$1" = "train" ]
then
    python3 -i main.py --data_path $DATA_PATH --model_path $MODEL_PATH --codebook_path $CODEBOOK_PATH --visualwords_path $VISUALWORDS_PATH --evaluation_path $EVALUATION_PATH --train
elif [ "$1" = "test" ]
then
    python3 main.py  --data_path $DATA_PATH --model_path $MODEL_PATH --evaluation_path $EVALUATION_PATH --descriptor $FEATURES_DESCRIPTOR --classifier $CLASSIFIER --train_method $TRAIN_METHOD --test
else
    echo "Incorrect option"
fi