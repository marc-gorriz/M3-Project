import argparse

from Input import Input
from KNN import KNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--classifier', type=str, default="knn")
    parser.add_argument('--train_method', type=str, default="kfold")
    parser.add_argument('--kfold_k', type=int, default=5)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_false', help='Flag to test or not.')

    args = parser.parse_args()

    InputData = Input(workingPath=args.data_path, nsamplesClass=30, train_method=args.train_method,
                      k=args.kfold_k, shuffle=True)

    # TODO: think about extend the code to other classification methods, features extractors ... maybe a switch?

    myKNN = KNN(nneighbors=100)

    if args.do_train:

        if args.train_method == 'kfold':

            # make K trainings, save the evaluation metrics and models, then decide the best model
            evaluation_metrics = []
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
                    predictions = myKNN.predict(validation_data, k_model)
                    # TODO: define the evaluation metrics class Evaluation, decide to plot, save, print ...
                    #model_evaluation = Evaluation.evaluation_method(predictions, validation_data['labels'])

                    model.append(k_model)
                    #evaluation_metrics.append(model_evaluation)

                else:
                    break

            # Decide the best model
            # TODO: implement a function in KNN to decide the best model based on evaluation_metrics
            model = myKNN.best_model(model, evaluation_metrics)

            # save model
            myKNN.save_model(model, args.model_path, 'kFold_model.sav')

        elif args.train_method == 'fixed':

            labeled_data = InputData.get_labeled_data()
            train_data = InputData.method_data_dictionary(labeled_data, 'train')
            validation_data = InputData.method_data_dictionary(labeled_data, 'validation')
            #del labeled_data

            # train model
            model = myKNN.train(train_data)

            # validate model
            predictions = myKNN.predict(validation_data['filenames'], model)
            # TODO: define the evaluation metrics class Evaluation, decide to plot, save, print ...
            #model_evaluation = Evaluation.evaluation_method(predictions, validation_data['labels'])

            # save model
            myKNN.save_model(model, args.model_path, 'fixed_model.sav')

        else:
            print("Invalid train method")


    elif args.do.test:

        test_data = InputData.method_data_dictionary(InputData.get_test_data(), 'test')

        # load model: in that case can be kFold or fixed model
        model = myKNN.load_model(args.model_path, 'kFold_model.sav')

        # Test the model
        predictions = myKNN.predict(test_data['filenames'], model)
        # TODO: define the evaluation metrics class Evaluation, decide to plot, save, print ...
        #model_evaluation = Evaluation.evaluation_method(predictions, test_data['labels'])
