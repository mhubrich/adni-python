from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dense, Dropout


def build_model(input_shape=(1, 8, 8, 8)):
	model = Sequential()
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', input_shape=input_shape))
        model.add(Dropout(0.2))
	model.add(Convolution3D(48, 3, 3, 3, activation='relu'))
        model.add(Dropout(0.2))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
        model.add(Dropout(0.2))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
        model.add(Dropout(0.2))
	model.add(Flatten())
	return model

