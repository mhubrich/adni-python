from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 10, 10, 10)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.001)))
	model.add(Flatten())
	return model

