import argparse
import time

import numpy as np

from Descriptors import SIFT, SURF, HOG
from Evaluation import Evaluation
from Input import Input
from KNN import KNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--evaluation_path', type=str, default="evaluation")
    parser.add_argument('--descriptor', type=str, default="surf")
    parser.add_argument('--classifier', type=str, default="knn")
    parser.add_argument('--train_method', type=str, default="kfold")
    parser.add_argument('--kfold_k', type=int, default=5)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_true', help='Flag to test or not.')

    args = parser.parse_args()

    InputData = Input(workingPath=args.data_path, nsamplesClass=80, train_method=args.train_method,
                      k=args.kfold_k, shuffle=True)

    # TODO: think about extend the code to other classification methods, features extractors ... maybe a switch?

    if args.descriptor == 'sift':
        features_descriptor = SIFT(nfeatures=100)

    elif args.descriptor == 'surf':
        features_descriptor = SURF(nOctaves=4, nOctaveLayers=2)

    elif args.descriptor == 'hog':
        features_descriptor = HOG()

    else:
        features_descriptor = None
        print('Invalid descriptor')

    myKNN = KNN(nneighbors=100, features_descriptor=features_descriptor)
    myEvaluation = Evaluation(evaluation_path=args.evaluation_path, save_plots=True)

    if args.do_train:

        start_time = time.time()

        if args.train_method == 'kfold':

            # make K trainings, save the evaluation metrics and models, then decide the best model
            evaluation_metrics = np.array([], dtype=float)
            model = []

            for k in range(args.kfold_k):

                labeled_data = InputData.get_labeled_data()
                train_data = InputData.method_data_dictionary(labeled_data, 'train')
                validation_data = InputData.method_data_dictionary(labeled_data, 'validation')
                # del labeled_data

                # train model
                k_model = myKNN.train(train_data)

                # validate model
                predictions = myKNN.predict(validation_data['filenames'], k_model, display=False)
                k_evaluation = myEvaluation.accuracy(validation_data['labels'], predictions, display=True)

                model.append(k_model)
                evaluation_metrics = np.hstack((evaluation_metrics, k_evaluation))


            # Decide the best model
            model = myEvaluation.best_model(evaluation_metrics, model)

            # save model
            myKNN.save_model(model, args.model_path)


        elif args.train_method == 'fixed':

            labeled_data = InputData.get_labeled_data()
            train_data = InputData.method_data_dictionary(labeled_data, 'train')
            validation_data = InputData.method_data_dictionary(labeled_data, 'validation')
            # del labeled_data

            # train model
            model = myKNN.train(train_data)

            # validate model
            predictions = myKNN.predict(validation_data['filenames'], model, display=False)
            myEvaluation.accuracy(validation_data['labels'], predictions, display=True)

            # save model
            myKNN.save_model(model, args.model_path)

        else:
            print("Invalid train method")

        print('Training finished: done in ' + str(time.time() - start_time) + ' secs')


    elif args.do_test:

        start_time = time.time()

        test_data = InputData.method_data_dictionary(InputData.get_test_data(), 'test')

        model = myKNN.load_model(args.model_path)

        # test model
        predictions = myKNN.predict(test_data['filenames'], model, display=True)
        myEvaluation.accuracy(test_data['labels'], predictions, display=True)
        myEvaluation.confusion_matrix(test_data['labels'], predictions, display=True)

        print('Test finished: done in ' + str(time.time() - start_time) + ' secs')
