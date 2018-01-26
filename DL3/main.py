import getpass
import matplotlib.pyplot as plt
import os
import sys
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Flatten, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from cnn_models import deep_model, CNNS_model
from constants import *

os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

train_idx = str(sys.argv[1])
output_path = global_path + "train" + train_idx + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

log_file = open(global_path + "log_file.txt", 'a')

if cnn_model == "deep_model":
    model = deep_model()
elif cnn_model == "cnns_model":
    model = CNNS_model()
else:
    print("Invalid model")
    model = None
    quit()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Data generation
train_datagen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=True,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=True,
                                   rotation_range=10,
                                   width_shift_range=0.,
                                   height_shift_range=0.,
                                   shear_range=0.2,
                                   zoom_range=0,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=None)

test_datagen = ImageDataGenerator(featurewise_center=False,
                                  samplewise_center=True,
                                  featurewise_std_normalization=False,
                                  samplewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(val_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# Training model
tensorboard = TensorBoard(log_dir=output_path, histogram_freq=0, write_graph=True, write_images=False)
model_checkpoint = ModelCheckpoint(filepath=output_path + 'weights.{epoch:02d}.hdf5', monitor='val_acc',
                                   verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', min_delta=early_delta, patience=early_patience, verbose=0, mode='auto')

history = model.fit_generator(train_generator,
                              samples_per_epoch=nb_train * augmentation_increment,  # data augmentation
                              nb_epoch=number_of_epoch,
                              validation_data=validation_generator,
                              nb_val_samples=nb_validation,
                              callbacks=[tensorboard, model_checkpoint, early_stop])

result = model.evaluate_generator(test_generator, nb_test, workers=12)
log_file.write("train" + train_idx + " " + str(result) + "\n")
log_file.close()
