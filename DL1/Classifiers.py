from keras.layers import Flatten, Dense, Reshape
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import numpy as np

import constants


class MLP:
    def __init__(self, output_path=None):
        self.output_path = output_path

    def build_mlp(self):
        input_size = constants.IMG_SIZE
        n_classes = len(constants.CLASSES)

        model = Sequential()

        model.add(Reshape((input_size * input_size * 3,), input_shape=(input_size, input_size, 3)))

        model.add(Dense(units=2048, activation='relu', name='hidden1'))
        model.add(Dense(units=1024, activation='relu', name='hidden2'))
        model.add(Dense(units=512, activation='relu', name='hidden3'))

        model.add(Dense(units=n_classes, activation='softmax'))

        return model

    def train_mlp(self, train_generator, validation_generator, summary=True, save_plots=True, save_model=True):
        model = self.build_mlp()
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        if summary: print(model.summary())
        if save_plots: plot_model(model, to_file=os.path.join(self.output_path, 'modelMLP.png'), show_shapes=True,
                                  show_layer_names=True)

        tensorboard = TensorBoard(log_dir=self.output_path, histogram_freq=0, write_graph=True, write_images=False)

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=constants.N_TRAIN // constants.BATCH_SIZE,
            epochs=constants.EPOCHS,
            validation_data=validation_generator,
            validation_steps=constants.N_TEST // constants.BATCH_SIZE,
            callbacks=[tensorboard])

        if save_model: model.save_weights(os.path.join(self.output_path, 'weights.h5'))

        if save_plots:
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(os.path.join(self.output_path, 'accuracy.jpg'))
            plt.close()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(os.path.join(self.output_path, 'loss.jpg'))

    def predict_mlp(self, generator, model):

        predictions = []
        labels = []

        for _ in range(constants.N_TEST // constants.BATCH_SIZE):
            image_batch, labels_batch = next(generator)
            predictions.append(np.argmax(model.predict(image_batch), axis=1))
            labels.append(np.argmax(labels_batch, axis=1))

        return np.hstack(predictions), np.hstack(labels)


    def load_model(self, model_path):
        assert os.path.exists(model_path), 'Invalid model path'
        return self.build_mlp().load_weights(model_path)


class SVM:
    def __init__(self, kernel='rbf', C=10, gamma=.0001, output_path=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.output_path = output_path

    def build_svm(self):
        return svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train_svm(self, features, labels, stdSlr_path=None, save_model=True):

        if stdSlr_path is None:
            stdSlr = StandardScaler().fit(features)
            with open(os.path.join(self.output_path, 'stdSlr.pkl'), 'wb') as filename:
                pickle.dump(stdSlr, filename)

        else:
            with open(os.path.join(stdSlr_path, 'stdSlr.pkl'), 'wb') as filename:
                stdSlr = pickle.load(filename)

        features = stdSlr.transform(features)
        model = self.build_svm().fit(features, labels)

        if save_model:
            with open(os.path.join(self.output_path, 'svm_model.pkl'), 'wb') as filename:
                pickle.dump(model, filename)

    def predict_svm(self, features, model, stdSlr_path):
        assert os.path.exists(stdSlr_path), 'Invalid stdSlr path'
        with open(os.path.join(stdSlr_path, 'stdSlr.pkl'), 'wb') as filename:
            stdSlr = pickle.load(filename)

        features = stdSlr.transform(features)
        return model.predict(features)

    def load_model(self, model_path):
        assert os.path.exists(model_path), 'Invalid model path'
        with open(model_path, 'wb') as filename:
            model = pickle.load(filename)
        return model

