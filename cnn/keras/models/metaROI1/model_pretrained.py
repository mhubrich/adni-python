from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 8, 8, 8)):
	l = 0.0001
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='ROI1_noise1'))
	model.add(Convolution3D(256, 5, 5, 5, activation='relu', W_regularizer=l2(l), name='ROI1_conv1'))
        model.add(Dropout(0.1, name='ROI1_dropout1'))
	model.add(Convolution3D(512, 3, 3, 3, activation='relu', W_regularizer=l2(l), name='ROI1_conv2'))
        model.add(Dropout(0.1, name='ROI1_dropout2'))
        model.add(Convolution3D(1024, 2, 2, 2, activation='relu', W_regularizer=l2(l), name='ROI1_conv3'))
        model.add(Dropout(0.1, name='ROI1_dropout3'))
	model.add(Flatten(name='ROI1_flatten1'))
        for l in model.layers:
            l.trainable = False
	return model

