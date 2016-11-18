from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 6, 6, 6)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='ROI2_noise1'))
	model.add(Convolution3D(512, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='ROI2_conv1'))
        model.add(Dropout(0.1, name='ROI2_dropout1'))
        model.add(Convolution3D(1024, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name='ROI2_conv2'))
        model.add(Dropout(0.1, name='ROI2_dropout2'))
	model.add(Flatten(name='ROI2_flatten1'))
        for l in model.layers:
            l.trainable = False
	return model

