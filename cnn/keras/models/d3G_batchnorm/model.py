from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization


def build_model(num_classes, input_shape=(1, 29, 29, 29)):
    model = Sequential()

    model.add(BatchNormalization(axis=1, input_shape=input_shape))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
