from keras.regularizers import l2
from keras.layers import Input, Dense, GaussianNoise, Flatten, Dropout, ZeroPadding3D, Convolution3D, UpSampling3D
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

    #x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Convolution3D(512, 2, 2, 2, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv8')(x)
    x = Dropout(do, name=name+'_dropout8')(x)
    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv9')(x)
    x = Dropout(do, name=name+'_dropout9')(x)
    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv10')(x)
    x = Dropout(do, name=name+'_dropout10')(x)
    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv11')(x)
    x = Dropout(do, name=name+'_dropout11')(x)
    x = ZeroPadding3D(padding=(2, 2, 2))(x)
    x = Convolution3D(64, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv12')(x)
    x = Dropout(do, name=name+'_dropout12')(x)
    x = ZeroPadding3D(padding=(2, 2, 2))(x)
    x = Convolution3D(32, 5, 5, 5, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv13')(x)
    x = Dropout(do, name=name+'_dropout13')(x)
    x = ZeroPadding3D(padding=(3, 3, 3))(x)
    x = Convolution3D(1, 7, 7, 7, activation='relu', W_regularizer=l2(0.0001), border_mode='same', name=name+'_conv14')(x)

    return Model(input=input, output=x)

