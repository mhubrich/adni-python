from keras.regularizers import l2
from keras.layers import Input, Dense, GaussianNoise, Convolution3D, Flatten, Dropout
from keras.models import Model


def build_model(input_shape=(1, 22, 22, 22), name='deepROI4'):
    name = name
    do = 1.0/14
    input = Input(shape=input_shape, name=name+'_input')
    x = GaussianNoise(0.001, name=name+'_noise1')(input)
    x = Convolution3D(32, 7, 7, 7, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv1')(x)
    x = Dropout(do, name=name+'_dropout1')(x)
    x = Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv2')(x)
    x = Dropout(do, name=name+'_dropout2')(x)
    x = Convolution3D(64, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv3')(x)
    x = Dropout(do, name=name+'_dropout3')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv4')(x)
    x = Dropout(do, name=name+'_dropout4')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv5')(x)
    x = Dropout(do, name=name+'_dropout5')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv6')(x)
    x = Dropout(do, name=name+'_dropout6')(x)
    x = Convolution3D(512, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), name=name+'_conv7')(x)
    x = Dropout(do, name=name+'_dropout7')(x)
    x = Flatten(name=name+'_flatten1')(x)

    return input, x

