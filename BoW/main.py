import argparse
import time
import os
import pickle

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
    parser.add_argument('--kfold_k', type=int, default=5)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_true', help='Flag to test or not.')
    parser.add_argument('--compute_features', dest='do_compute_features', action='store_true',
                        help='Flag to compute_features or not.')

    args = parser.parse_args()

    InputData = Input(workingPath=args.data_path, nsamplesClass=80, shuffle=False)

    # TODO: think about extend the code to other classification methods, features extractors ... maybe a switch?

    if args.descriptor == 'bow':
        sift_descriptors = SIFT(nfeatures=300, type='DENSE', step=5)
        bow_descriptor = BOW(k=512)

    else:
        sift_descriptors = None
        bow_descriptor = None
        print('Invalid descriptor')

    mySVM = SVM(kernel='rbf', C=1, gamma=.002)
    myEvaluation = Evaluation(evaluation_path=args.evaluation_path, save_plots=True)

    if args.do_train:

        start_time = time.time()
        train_data = InputData.get_labeled_data()

        if args.do_compute_features:
            train_descriptors, train_idx = sift_descriptors.extract_features_simple(data_dictionary=train_data)
            codebook = bow_descriptor.compute_codebook(train_descriptors)
            bow_descriptor.save(codebook, os.path.join(args.codebook_path, 'codebook.pkl'), 'codebook')
            train_visual_words = bow_descriptor.extract_features(train_descriptors, codebook)
            bow_descriptor.save(train_visual_words, os.path.join(args.visualwords_path,
                                'train_visual_words.npy'), 'visualwords')

        else:
            train_visual_words = bow_descriptor.load(os.path.join(args.visualwords_path,
                                'train_visual_words.npy'), 'visualwords')


        # cross validation
        best_params_svm = mySVM.cross_validation(train_visual_words, train_data)
        del mySVM
        mySVM = SVM(kernel=best_params_svm['kernel'], C=best_params_svm['C'], gamma=best_params_svm['gamma'])

        # train model
        model = mySVM.train(train_visual_words, train_data)

        # save model
        mySVM.save_model(model, args.model_path)

        # hardcode
        with open("../../Lab2-BoW/test1/best_params_svm.pkl", 'wb') as file:
            pickle.dump(best_params_svm, file)


    elif args.do_test:

        start_time = time.time()

        # hardcode
        with open("../../Lab2-BoW/test1/best_params_svm.pkl", 'rb') as file:
            best_params_svm = pickle.load(file)

        # loading model and the best parameters found with cross validation
        del mySVM
        mySVM = SVM(kernel=best_params_svm['kernel'], C=best_params_svm['C'], gamma=best_params_svm['gamma'])
        model = mySVM.load_model(args.model_path)

        test_data = InputData.get_test_data()

        if args.do_compute_features:
            test_descriptors, test_idx = sift_descriptors.extract_features_simple(data_dictionary=test_data)
            codebook = bow_descriptor.load(os.path.join(args.codebook_path, 'codebook.pkl'), 'codebook')
            test_visual_words = bow_descriptor.extract_features(test_descriptors, codebook)
            bow_descriptor.save(test_visual_words, os.path.join(args.visualwords_path,
                                'test_visual_words.npy'), 'visualwords')

        else:
            test_visual_words = bow_descriptor.load(os.path.join(args.visualwords_path,
                                'test_visual_words.npy'), 'visualwords')


        predictions = mySVM.predict(model, test_visual_words, test_data)
        myEvaluation.accuracy(test_data['labels'], predictions, display=True)
        myEvaluation.confusion_matrix(test_data['labels'], predictions, display=True)

        print('Test finished: done in ' + str(time.time() - start_time) + ' secs')
