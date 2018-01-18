from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Conv2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
#from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import getpass
import os
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

train_data_dir='../../Databases/MIT_split_reduced/train'
val_data_dir='../../Databases/MIT_split_reduced/validation'
test_data_dir='../../Databases/MIT_split_reduced/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=20


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x
    
# create the base pre-trained model
base_model = VGG16(weights='imagenet')
base_model.summary()
x = base_model.get_layer('block4_pool').output
x = Conv2D(64,1, activation='relu')(x)
x = Flatten(name='flatten')(x)
#x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
#x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=x)
#plot(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

for layer in model.layers:
    print (layer.name, layer.trainable)

#model.summary()

#preprocessing_function=preprocess_input,

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

#train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
#datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch = 400,
        nb_epoch=number_of_epoch,
        validation_data=test_generator,
        nb_val_samples=120)

model.save_weights('weights.h5')

result = model.evaluate_generator(test_generator) #, val_samples=807)
print (result)


# list all data in history

if False:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
