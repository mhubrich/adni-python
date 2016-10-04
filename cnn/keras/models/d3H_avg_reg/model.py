from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import AveragePooling3D
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(num_classes, input_shape=(1, 58, 58, 58)):
    model = Sequential()

    model.add(GaussianNoise(sigma=0.01, input_shape=input_shape))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))

    model.add(GaussianNoise(sigma=0.01, input_shape=input_shape))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.001)))

    model.add(GaussianNoise(sigma=0.01))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))

    model.add(GaussianNoise(sigma=0.01))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', W_regularizer=l2(0.001)))

    return model

