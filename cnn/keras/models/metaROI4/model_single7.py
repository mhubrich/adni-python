from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 13, 13, 13)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu'))
        model.add(Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution3D(48, 3, 3, 3, activation='relu'))
        model.add(Convolution3D(48, 3, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
        model.add(Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
        model.add(Convolution3D(256, 3, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
	model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

	return model

