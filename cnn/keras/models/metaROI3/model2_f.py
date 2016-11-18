from keras.layers import Input, Convolution3D, Flatten
from keras.models import Model


def build_model(input_shape=(1, 10, 10, 10)):
	input = Input(shape=input_shape)
	x = Convolution3D(32, 3, 3, 3, activation='relu')(input)
	x = Convolution3D(48, 3, 3, 3, activation='relu')(x)
	x = Convolution3D(64, 3, 3, 3, activation='relu')(x)
	x = Convolution3D(128, 3, 3, 3, activation='relu')(x)
	out = Flatten()(x)
        return input, out

