from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten
from keras.layers.noise import GaussianNoise


def build_model(input_shape=(1, 6, 6, 6)):
	model = Sequential()
        model.add(GaussianNoise(0.5, input_shape=input_shape))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution3D(48, 3, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
	model.add(Flatten())
	return model

