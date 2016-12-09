from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling3D


def build_model(input_shape=(1, 20, 20, 20)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='meanROI1_4_noise1'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_conv1'))
        model.add(Dropout(1.0/10, name='meanROI1_4_dropout1'))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_conv2'))
        model.add(Dropout(1.0/10, name='meanROI1_4_dropout2'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_conv3'))
        model.add(Dropout(1.0/10, name='meanROI1_4_dropout3'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_conv4'))
        model.add(Dropout(1.0/10, name='meanROI1_4_dropout4'))
	model.add(Convolution3D(256, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_conv5'))
        model.add(Dropout(1.0/10, name='meanROI1_4_dropout5'))
	model.add(Flatten(name='meanROI1_4_flatten1'))

        model.add(Dense(256, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_dense1'))
        model.add(Dropout(0.25, name='meanROI1_4_dropout6'))
        model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001), name='meanROI1_4_dense2'))
        model.add(Dropout(0.25, name='meanROI1_4_dropout7'))
        model.add(Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), name='meanROI1_4_dense3'))
	return model

