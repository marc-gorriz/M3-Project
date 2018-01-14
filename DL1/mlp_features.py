import os
import cPickle as pickle 
from keras.layers import Flatten, Dense, Reshape
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import cluster
from sklearn.feature_extraction.image import extract_patches_2d

from utils import *
import constants


def get_model_layer(model):
    print(model)
    print(model.layers)
    return Model(input=model.input, output=model.get_layer(constants.LAYER_FEATURES_GENERATOR).output)

def batch_single_features(batch_images, model):
    return model.predict(batch_images)


def image_multiple_features(image, img_idx, model):
    patches = extract_patches_2d(image, (constants.PATCH_SIZE, constants.PATCH_SIZE), max_patches=constants.N_PATCHES)
    features_idx = np.array([img_idx] * len(patches))
    return model.predict(patches), features_idx


def extract_layer_features(generator, n_images, model):

    features_map = []
    labels = []

    model_layer = get_model_layer(model)

    for _ in range(n_images // constants.BATCH_SIZE):
        (batch_images, batch_labels) = next(generator)
        features = batch_single_features(batch_images, model_layer)

        features_map.append(features)
        labels.append(np.argmax(batch_labels, axis=1))

    return np.vstack(features_map), np.hstack(labels)


def extract_visual_words(generator, n_images, model, k, codebook_path=None, output_path=None):
    features_map = []
    idx_map = []
    labels = []

    model_layer = get_model_layer(model)

    for step in range(n_images // constants.BATCH_SIZE):
        (batch_images, batch_labels) = next(generator)
        labels.append(np.argmax(batch_labels, axis=1))
        for idx, image in enumerate(batch_images):
            img_idx = step * constants.BATCH_SIZE + idx
            features, features_idx = image_multiple_features(image, img_idx, model_layer)
            features_map.append(features)
            idx_map.append(features_idx)

    features_map = np.vstack(features_map)
    idx_map = np.hstack(idx_map)

    if codebook_path is None:
        # Compute codebook
        codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,
                                           compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(features_map)

        
        assert output_path is not None
        with open(os.path.join(output_path, 'codebook.pkl'), 'wb') as filename:
            pickle.dump(codebook, filename)

    else:
        with open(os.path.join(codebook_path, 'codebook.pkl'), 'rb') as filename:
            codebook = pickle.load(filename)
    
    labels = np.hstack(labels)
    # compute visual words
    codebook_predictions = codebook.predict(features_map)
    visual_words = np.zeros((len(labels), k), dtype=np.float32)
    for idx in range(0, len(labels)):
        visual_words[idx, :] = np.bincount(codebook_predictions[idx_map == idx], minlength=k)

    return visual_words, labels
