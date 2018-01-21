import getpass
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Flatten, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

train_data_dir = '../../Databases/MIT_split_reduced/train'
val_data_dir = '../../Databases/MIT_split_reduced/validation'
test_data_dir = '../../Databases/MIT_split_reduced/test'
img_width = 224
img_height = 224
batch_size = 32
number_of_epoch = 15

save_weights = True
preview_augmentation = False
plot_evaluation = False

# create the base pre-trained model
base_model = VGG16(weights='imagenet')
x = base_model.get_layer('block4_pool').output
x = Conv2D(64, 1, activation='relu')(x)
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=x)

adadelta = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

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

if preview_augmentation:
    # Preview resultant images:
    img = load_img(train_data_dir + '/mountain/land132.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    i = 0
    for batch in train_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='image', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely

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

history = model.fit_generator(train_generator,
                              samples_per_epoch=400 * 20,  # data augmentation
                              nb_epoch=number_of_epoch,
                              validation_data=test_generator,
                              nb_val_samples=120)  # data augmentation

if save_weights: model.save_weights('weights.h5')

result = model.evaluate_generator(test_generator, 807, workers=12)  # , val_samples=807)
print("result: " + str(result))

if plot_evaluation:
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