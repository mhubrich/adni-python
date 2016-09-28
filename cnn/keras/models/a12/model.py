from cnn.keras.models.sequential_multi_threading import SequentialMultiThreading
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout


def build_model(num_classes, input_shape=(3, 96, 96)):
    model = SequentialMultiThreading()

    model.add(Convolution2D(32, 7, 7, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(32, 7, 7, activation='relu'))
    model.add(Convolution2D(32, 7, 7, activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, 7, 7, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, 7, 7, activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))

    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(Convolution2D(128, 5, 5, activation='relu'))

    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
