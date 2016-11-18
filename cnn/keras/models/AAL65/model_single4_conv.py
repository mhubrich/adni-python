from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 21, 21, 21)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='AAL65_noise1'))
	model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv1'))
        model.add(Dropout(1.0/14, name='AAL65_dropout1'))
	model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv2'))
        model.add(Dropout(1.0/14, name='AAL65_dropout2'))
        model.add(Convolution3D(64, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv3'))
        model.add(Dropout(1.0/14, name='AAL65_dropout3'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv4'))
        model.add(Dropout(1.0/14, name='AAL65_dropout4'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv5'))
        model.add(Dropout(1.0/14, name='AAL65_dropout5'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv6'))
        model.add(Dropout(1.0/14, name='AAL65_dropout6'))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL65_conv7'))
        model.add(Dropout(1.0/14, name='AAL65_dropout7'))
	model.add(Flatten(name='AAL65_flatten1'))
	return model

