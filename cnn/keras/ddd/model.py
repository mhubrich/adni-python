from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D


def build_model(num_classes):
    model = Sequential()
    model.add(ZeroPadding3D((1, 1, 1), input_shape=(1, 160, 160, 96)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(1024, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(ZeroPadding3D((1, 1, 1)))
    model.add(Convolution3D(1024, 3, 3, 3, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
