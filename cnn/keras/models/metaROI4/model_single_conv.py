from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 13, 13, 13)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='ROI4_noise1'))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv1'))
        model.add(Dropout(0.1, name='ROI4_dropout1'))
	model.add(Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv2'))
        model.add(Dropout(0.1, name='ROI4_dropout2'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv3'))
        model.add(Dropout(0.1, name='ROI4_dropout3'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv4'))
        model.add(Dropout(0.1, name='ROI4_dropout4'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv5'))
        model.add(Dropout(0.1, name='ROI4_dropout5'))
	model.add(Convolution3D(512, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='ROI4_conv6'))
        model.add(Dropout(0.1, name='ROI4_dropout6'))
	model.add(Flatten(name='ROI4_flatten1'))
        for l in model.layers:
            l.trainable = True
        return model

