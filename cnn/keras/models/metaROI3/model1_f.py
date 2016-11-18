from keras.layers import Input, Convolution3D, Flatten, merge, Dense, Dropout
from keras.models import Model

from cnn.keras.metaROI_autoencoder.model_input import build_model as mod_auto


def build_model(input_shape=(1, 10, 10, 10)):
	input1 = Input(shape=input_shape)
	x = Convolution3D(32, 3, 3, 3, activation='relu')(input1)
	x = Convolution3D(48, 3, 3, 3, activation='relu')(x)
	x = Convolution3D(64, 3, 3, 3, activation='relu')(x)
	x = Convolution3D(128, 3, 3, 3, activation='relu')(x)
	out1 = Flatten()(x)

        input2 = mod_auto()

        merged = merge([out1, input2], mode='concat')
        x = Dense(512, activation='relu')(merged)
        x = Dropout(0.1)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.1)(x)
        out2 = Dense(512, activation='relu')(x)

        return input1, input2, out2
