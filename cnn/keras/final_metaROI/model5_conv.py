from keras.regularizers import l2
from keras.layers import Input, Dense, GaussianNoise, Convolution3D, Flatten, Dropout
from keras.models import Model


def build_model(input_shape=(1, 5, 5, 5)):
    name = 'metaROI5'
    do = 1.0/4
    input = Input(shape=input_shape, name=name+'_input')
    x = GaussianNoise(0.001, name=name+'_noise1')(input)
    x = Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv1')(x)
    x = Dropout(do, name=name+'_dropout1')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv2')(x)
    x = Dropout(do, name=name+'_dropout2')(x)
    x = Flatten(name=name+'_flatten1')(x)

    return input, x

