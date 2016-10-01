from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.layers.core import Flatten, Dense, Dropout


def build_model(num_classes, input_shape=(3, 96, 96)):
    model = Sequential()

    model.add(Convolution2D(32, 7, 7, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(32, 7, 7, activation='relu'))
    model.add(Convolution2D(32, 7, 7, activation='relu'))
    model.add(Convolution2D(32, 7, 7, activation='relu'))  # new

    model.add(Convolution2D(64, 7, 7, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, 7, 7, activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))  # new

    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(Convolution2D(128, 5, 5, activation='relu'))  # new

    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # new

    model.add(Convolution2D(num_classes, 1, 1, activation='sigmoid'))

    model.add(Flatten())

    return model
