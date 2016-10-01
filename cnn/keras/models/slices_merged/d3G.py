from cnn.keras.models.sequential_multi_threading import SequentialMultiThreading
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten


def build_model(num_classes, input_shape=(1, 29, 29, 29)):
    model = SequentialMultiThreading()

    model.add(Convolution3D(32, 5, 5, 5, activation='relu', input_shape=input_shape))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))

    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu'))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
    model.add(Convolution3D(64, 1, 1, 1, activation='relu'))

    model.add(Flatten())

    return model
