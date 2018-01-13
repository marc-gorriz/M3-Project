import sys
from keras.preprocessing.image import ImageDataGenerator

import Evaluation
import constants
import mlp_features
from Classifiers import MLP, SVM
from utils import *

OUTPUT_DIR = '../../DL1-OUTPUT/train/' + str(sys.argv[1])

if not os.path.exists(constants.DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory ' + constants.DATASET_DIR + ' do not exists!\n')
    quit()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Data generation
if constants.FEATURES_EXTRACTOR is 'bow' and constants.CLASSIFIER is 'svm': constants.IMG_SIZE = constants.DATA_SIZE

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR + '/train', target_size=(constants.IMG_SIZE, constants.IMG_SIZE),
    batch_size=constants.BATCH_SIZE, classes=constants.CLASSES, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR + '/test', target_size=(constants.IMG_SIZE, constants.IMG_SIZE),
    batch_size=constants.BATCH_SIZE, classes=constants.CLASSES, class_mode='categorical')

if constants.DO_TRAIN:

    if constants.CLASSIFIER is 'svm':

        # Features extraction
        features_model = MLP().load_model(constants.FEATURES_MODEL)

        if constants.FEATURES_EXTRACTOR is 'simple':
            train_features, train_labels = mlp_features.extract_layer_features(generator=train_generator,
                                                                               n_images=constants.N_TRAIN,
                                                                               model=features_model)
        elif constants.FEATURES_EXTRACTOR is 'bow':
            train_features, train_labels = mlp_features.extract_visual_words(generator=train_generator,
                                                                             n_images=constants.N_TRAIN,
                                                                             model=features_model,
                                                                             k=constants.BOW_K,
                                                                             codebook_path=constants.CODEBOOK_PATH)
        else:
            print('Invalid features extractor')
            train_features = None
            train_labels = None
            quit()

        # SVM training
        SVM(output_path=OUTPUT_DIR).train_svm(features=train_features,
                                              labels=train_labels,
                                              stdSlr_path=constants.STDSLR_PATH,
                                              save_model=True)

    elif constants.CLASSIFIER is 'mlp':
        MLP(output_path=OUTPUT_DIR).train_mlp(train_generator=train_generator,
                                              validation_generator=test_generator,
                                              summary=True, save_plots=True, save_model=True)
    else:
        print('Invalid classifier')
        quit()

if constants.DO_TEST:

    if constants.CLASSIFIER is 'svm':

        # Features extraction
        features_model = MLP().load_model(constants.FEATURES_MODEL)

        if constants.FEATURES_EXTRACTOR is 'simple':
            test_features, test_labels = mlp_features.extract_layer_features(generator=test_generator,
                                                                             n_images=constants.N_TEST,
                                                                             model=features_model)
        elif constants.FEATURES_EXTRACTOR is 'bow':
            test_features, test_labels = mlp_features.extract_visual_words(generator=test_generator,
                                                                           n_images=constants.N_TEST,
                                                                           model=features_model,
                                                                           k=constants.BOW_K,
                                                                           codebook_path=constants.CODEBOOK_PATH)
        else:
            print('Invalid features extractor')
            test_features = None
            test_labels = None
            quit()

        # Getting SVM predictions
        svm_classifier = SVM()
        model_svm = svm_classifier.load_model(os.path.join(OUTPUT_DIR, 'svm_model.pkl'))
        stdSlr_path = os.path.join(OUTPUT_DIR, 'stdSlr.pkl')
        predictions = svm_classifier.predict_svm(test_features, model_svm, stdSlr_path=stdSlr_path)

    elif constants.CLASSIFIER is 'mlp':
        mlp_classifier = MLP()
        model_mlp = mlp_classifier.load_model(os.path.join(OUTPUT_DIR, 'weights.h5'))
        predictions, test_labels = mlp_classifier.predict_mlp(test_generator, model_mlp)

    else:
        print('Invalid classifier')
        test_labels = None
        predictions = None
        quit()

    # Overall evaluations
    Evaluation.accuracy(ground_truth=labels, predictions=predictions, display=True)
    Evaluation.confusion_matrix(ground_truth=labels, predictions=predictions, display=True, output_path=OUTPUT_DIR)
