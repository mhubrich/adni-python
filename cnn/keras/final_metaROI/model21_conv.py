from keras.regularizers import l2
from keras.layers import Input, Dense, GaussianNoise, Convolution3D, Flatten, Dropout
from keras.models import Model


def build_model(input_shape=(1, 8, 8, 8), trainable=True):
    name = 'metaROI1'
    do = 1.0/8
    input = Input(shape=input_shape, name=name+'_input')
    x = GaussianNoise(0.001, name=name+'_noise1')(input)
    x = Convolution3D(32, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv1', trainable=trainable)(x)
    x = Dropout(do, name=name+'_dropout1')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv2', trainable=trainable)(x)
    x = Dropout(do, name=name+'_dropout2')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv3', trainable=trainable)(x)
    x = Dropout(do, name=name+'_dropout3')(x)
    x = Convolution3D(256, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv4', trainable=trainable)(x)
    x = Dropout(do, name=name+'_dropout4')(x)
    x = Flatten(name=name+'_flatten1')(x)
    return input, x
