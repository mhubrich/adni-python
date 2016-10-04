from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, ZeroPadding3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling3D


def build_model(num_classes, input_shape=(1, 54, 54, 54)):
    model = Sequential()

    model.add(ZeroPadding3D(padding=(2, 2, 2), input_shape=input_shape))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), input_shape=input_shape))

    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))

    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    #model.add(Convolution3D(128, 3, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
