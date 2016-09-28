from cnn.keras.models.sequential_multi_threading import SequentialMultiThreading
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.regularizers import l2


def build_model(num_classes, input_shape=(1, 29, 29, 29)):
    model = SequentialMultiThreading()

    model.add(Convolution3D(32, 5, 5, 5, activation='relu', input_shape=input_shape, W_regularizer=l2(0.001)))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.001)))

    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  # , W_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
