from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, ZeroPadding3D
from keras.layers.core import Flatten, Dense, Dropout


def build_model(num_classes, input_shape=(1, 20, 20, 20)):
    model = Sequential()

    model.add(ZeroPadding3D(padding=(3, 3, 3), input_shape=input_shape))
    model.add(Convolution3D(16, 7, 7, 7, activation='relu'))
    model.add(ZeroPadding3D(padding=(3, 3, 3)))
    model.add(Convolution3D(16, 7, 7, 7, activation='relu'))

    model.add(Convolution3D(16, 5, 5, 5, activation='relu'))
    model.add(ZeroPadding3D(padding=(2, 2, 2)))
    model.add(Convolution3D(16, 5, 5, 5, activation='relu'))

    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(ZeroPadding3D(padding=(2, 2, 2)))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))

    model.add(Convolution3D(64, 5, 5, 5, activation='relu'))
    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Convolution3D(64, 5, 5, 5, activation='relu'))

    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))  # new, requires input_shape=(1, 21, 21, 21) now

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
