import argparse
import numpy as np
import time

from Input import Input
from KNN import KNN
from Evaluation import Evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--evaluation_path', type=str, default="evaluation")
    parser.add_argument('--classifier', type=str, default="knn")
    parser.add_argument('--train_method', type=str, default="kfold")
    parser.add_argument('--kfold_k', type=int, default=5)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_true', help='Flag to test or not.')

    args = parser.parse_args()

    InputData = Input(workingPath=args.data_path, nsamplesClass=30, train_method=args.train_method,
                      k=args.kfold_k, shuffle=True)

    # TODO: think about extend the code to other classification methods, features extractors ... maybe a switch?

    myKNN = KNN(nneighbors=100)
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
                #del labeled_data

                if train_data is not False:

                    # train model
                    k_model = myKNN.train(train_data)

                    # validate model
                    predictions = myKNN.predict(validation_data['filenames'], k_model)
                    k_evaluation = myEvaluation.accuracy(validation_data['labels'], predictions, display=True)

                    model.append(k_model)
                    evaluation_metrics = np.hstack((evaluation_metrics, k_evaluation))

                else:
                    break

            # Decide the best model
            model = myEvaluation.best_model(evaluation_metrics, model)

            # save model
            myKNN.save_model(model, args.model_path, args.train_method + '_model.pkl')


        elif args.train_method == 'fixed':

            labeled_data = InputData.get_labeled_data()
            train_data = InputData.method_data_dictionary(labeled_data, 'train')
            validation_data = InputData.method_data_dictionary(labeled_data, 'validation')
            #del labeled_data

            # train model
            model = myKNN.train(train_data)

            # validate model
            predictions = myKNN.predict(validation_data['filenames'], model)
            myEvaluation.accuracy(validation_data['labels'], predictions, display=True)

            # save model
            myKNN.save_model(model, args.model_path, args.train_method + '_model.pkl')

        else:
            print("Invalid train method")

        print('Training finished: done in ' + str(time.time() - start_time) + ' secs')


    elif args.do_test:

        start_time = time.time()

        test_data = InputData.method_data_dictionary(InputData.get_test_data(), 'test')

        model = myKNN.load_model(args.model_path, args.train_method + '_model.pkl')

        # test model
        predictions = myKNN.predict(test_data['filenames'], model)
        myEvaluation.accuracy(test_data['labels'], predictions, display=True)
        myEvaluation.confusion_matrix(test_data['labels'], predictions, display=True)

        print('Test finished: done in ' + str(time.time() - start_time) + ' secs')
