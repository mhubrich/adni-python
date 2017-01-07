from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, UpSampling3D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2


def build_model(input_shape=(1, 22, 22, 22)):
        name = 'AVG444'
        do = 1.0/14
	model = Sequential()

        model.add(GaussianNoise(0.001, input_shape=input_shape, name=name+'_noise1'))
	model.add(Convolution3D(32, 7, 7, 7, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv1'))
        model.add(Dropout(do, name=name+'_dropout1'))
	model.add(Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv2'))
        model.add(Dropout(do, name=name+'_dropout2'))
        model.add(Convolution3D(64, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv3'))
        model.add(Dropout(do, name=name+'_dropout3'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv4'))
        model.add(Dropout(do, name=name+'_dropout4'))
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv5'))
        model.add(Dropout(do, name=name+'_dropout5'))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv6'))
        model.add(Dropout(do, name=name+'_dropout6'))
        model.add(Convolution3D(512, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv7'))
        model.add(Dropout(do, name=name+'_dropout7'))

        name = 'AVG444_decoder'
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling1'))
        model.add(Convolution3D(256, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv8'))
        model.add(Dropout(do, name=name+'_dropout8'))
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling2'))
        model.add(Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv9'))
        model.add(Dropout(do, name=name+'_dropout9'))
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling3'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv10'))
        model.add(Dropout(do, name=name+'_dropout10'))
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling4'))
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv11'))
        model.add(Dropout(do, name=name+'_dropout11'))
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling5'))
        model.add(Convolution3D(32, 7, 7, 7, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv12'))
        model.add(Dropout(do, name=name+'_dropout12'))
        model.add(UpSampling3D(size=(2,2,2), name=name+'_upsampling6'))
        model.add(Convolution3D(32, 7, 7, 7, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv13'))
        model.add(Dropout(do, name=name+'_dropout13'))

	return model

