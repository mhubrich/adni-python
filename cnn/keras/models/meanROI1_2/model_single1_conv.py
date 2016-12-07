from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling3D


def build_model(input_shape=(1, 23, 23, 23)):
        name = 'meanROI1_2'
        do = 0.1
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name=name+'_noise1'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv1'))
        model.add(Dropout(do, name=name+'_dropout1'))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv2'))
        model.add(Dropout(do, name=name+'_dropout2'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv3'))
        model.add(Dropout(do, name=name+'_dropout3'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv4'))
        model.add(Dropout(do, name=name+'_dropout4'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv5'))
        model.add(Dropout(do, name=name+'_dropout5'))
	model.add(Flatten(name=name+'_flatten1'))
	return model

