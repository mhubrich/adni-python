from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Merge

from cnn.keras.metaROItest.diff_input import build_model as mod_diff


def build_model(input_shape=(1, 10, 10, 10)):
	model = Sequential()
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', input_shape=input_shape))
	model.add(Convolution3D(48, 3, 3, 3, activation='relu'))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
	model.add(Flatten())

        model2 = mod_diff()

        merged = Merge([model, model2], mode='concat')
        mod = Sequential()
        mod.add(merged)

	mod.add(Dense(1024, activation='relu'))
        mod.add(Dropout(0.1))
        mod.add(Dense(1024, activation='relu'))
        mod.add(Dropout(0.1))
	return mod
