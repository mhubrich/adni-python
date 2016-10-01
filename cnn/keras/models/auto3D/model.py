from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, UpSampling3D, MaxPooling3D


def build_model(input_shape=(1, 96, 96, 96)):
    model = Sequential()

    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same'))

    model.add(UpSampling3D(size=(2, 2, 2)))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution3D(1, 3, 3, 3, activation='sigmoid', border_mode='same'))
    
    return model
