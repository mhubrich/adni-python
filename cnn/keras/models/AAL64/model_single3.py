from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 18, 18, 18)):
	model = Sequential()
        model.add(GaussianNoise(0.001, input_shape=input_shape, name='AAL64_noise1'))
	model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv1'))
        model.add(Dropout(1.0/12, name='AAL64_dropout1'))
	model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv2'))
        model.add(Dropout(1.0/12, name='AAL64_dropout2'))
        model.add(Convolution3D(64, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv3'))
        model.add(Dropout(1.0/12, name='AAL64_dropout3'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv4'))
        model.add(Dropout(1.0/12, name='AAL64_dropout4'))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv5'))
        model.add(Dropout(1.0/12, name='AAL64_dropout5'))
        model.add(Convolution3D(512, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name='AAL64_conv6'))
        model.add(Dropout(1.0/12, name='AAL64_dropout6'))
	model.add(Flatten(name='AAL64_flatten1'))

        model.add(Dense(256, activation='relu', W_regularizer=l2(0.0001), name='AAL64_dense1'))
        model.add(Dropout(0.2, name='AAL64_dropout7'))
        model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001), name='AAL64_dense2'))
        model.add(Dropout(0.2, name='AAL64_dropout8'))
        model.add(Dense(2, activation='softmax', W_regularizer=l2(0.0001), name='AAL64_dense3'))

	return model

