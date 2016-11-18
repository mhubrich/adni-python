from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2

def build_model(input_shape=(1, 5, 5, 5)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='ROI5_noise1'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI5_conv1'))
        model.add(Dropout(0.1, name='ROI5_dropout1'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI5_conv2'))
        model.add(Dropout(0.1, name='ROI5_dropout2'))
	model.add(Flatten(name='ROI5_flatten1'))

        model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001), name='ROI5_dense1'))
        model.add(Dropout(0.2, name='ROI5_dropout3'))
        model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001), name='ROI5_dense2'))
        model.add(Dropout(0.2, name='ROI5_dropout4'))
        model.add(Dense(2, activation='softmax', W_regularizer=l2(0.0001), name='ROI5_dense3'))

	return model

