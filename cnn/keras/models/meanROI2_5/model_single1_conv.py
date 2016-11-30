from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling3D


def build_model(input_shape=(1, 21, 21, 21)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='meanROI2_5_noise1'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI2_5_conv1'))
        model.add(Dropout(1.0/10, name='meanROI2_5_dropout1'))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI2_5_conv2'))
        model.add(Dropout(1.0/10, name='meanROI2_5_dropout2'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI2_5_conv3'))
        model.add(Dropout(1.0/10, name='meanROI2_5_dropout3'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI2_5_conv4'))
        model.add(Dropout(1.0/10, name='meanROI2_5_dropout4'))
	model.add(Convolution3D(256, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name='meanROI2_5_conv5'))
        model.add(Dropout(1.0/10, name='meanROI2_5_dropout5'))
	model.add(Flatten(name='meanROI2_5_flatten1'))
	return model

