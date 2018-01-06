import argparse
import time
import os

import numpy as np

from Descriptors import SIFT, BOW
from Evaluation import Evaluation
from Input import Input
from Classifiers import SVM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--codebook_path', type=str, default="codebook")
    parser.add_argument('--visualwords_path', type=str, default="visualwords_path")
    parser.add_argument('--evaluation_path', type=str, default="evaluation")
    parser.add_argument('--descriptor', type=str, default="bow")
    parser.add_argument('--classifier', type=str, default="svm")
    parser.add_argument('--train_method', type=str, default="kfold")
    parser.add_argument('--kfold_k', type=int, default=5)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_true', help='Flag to test or not.')
    parser.add_argument('--compute_features', dest='do_compute_features', action='store_true',
                        help='Flag to compute_features or not.')

    args = parser.parse_args()

    InputData = Input(workingPath=args.data_path, nsamplesClass=80, train_method=args.train_method,
                      k=args.kfold_k, shuffle=True)

    # TODO: think about extend the code to other classification methods, features extractors ... maybe a switch?

    if args.descriptor == 'bow':
        sift_descriptors = SIFT(nfeatures=100)
        bow_descriptor = BOW(k=512)

    else:
        sift_descriptors = None
        bow_descriptor = None
        print('Invalid descriptor')

    mySVM = SVM(kernel='rbf', C=1, gamma=.002)
    myEvaluation = Evaluation(evaluation_path=args.evaluation_path, save_plots=True)

    if args.do_train:

        start_time = time.time()
        labeled_data = InputData.get_labeled_data()
        train_data = InputData.method_data_dictionary(labeled_data, 'train')
        validation_data = InputData.method_data_dictionary(labeled_data, 'validation')

        if args.do_compute_features:
            train_descriptors = sift_descriptors.extract_features_simple(data_dictionary=train_data)
            train_codebook = bow_descriptor.compute_codebook(train_descriptors)
            bow_descriptor.save(train_codebook, os.path.join(args.codebook_path, 'train_codebook.pkl'), 'codebook')
            train_visual_words = bow_descriptor.extract_features(train_descriptors, train_codebook)
            bow_descriptor.save(train_visual_words, os.path.join(args.visualwords_path,
                                'train_visual_words.npy'), 'visualwords')

        else:
            # codebook = bow_descriptor.load(args.codebook_path, 'codebook')
            train_visual_words = bow_descriptor.load(os.path.join(args.visualwords_path,
                                'train_visual_words.npy'), 'visualwords')

        # train model
        model = mySVM.train(train_visual_words, train_data)
        # predictions = [classes_names[i] for i in model.predict(visual_words)]







    elif args.do_test:

        start_time = time.time()

        test_data = InputData.method_data_dictionary(InputData.get_test_data(), 'test')

        model = myKNN.load_model(args.model_path)

        # test model
        predictions = myKNN.predict(test_data['filenames'], model, display=True)
        myEvaluation.accuracy(test_data['labels'], predictions, display=True)
        myEvaluation.confusion_matrix(test_data['labels'], predictions, display=True)

        print('Test finished: done in ' + str(time.time() - start_time) + ' secs')
