from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 10, 10, 10)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='ROI3_noise1'))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI3_conv1'))
        model.add(Dropout(0.1, name='ROI3_dropout1'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI3_conv2'))
        model.add(Dropout(0.1, name='ROI3_dropout2'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI3_conv3'))
        model.add(Dropout(0.1, name='ROI3_dropout3'))
        model.add(Convolution3D(512, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name='ROI3_conv4'))
        model.add(Dropout(0.1, name='ROI3_dropout4'))
	model.add(Flatten(name='ROI3_flatten1'))
        for l in model.layers:
            l.trainable = True
        return model

