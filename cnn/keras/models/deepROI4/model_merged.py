from keras.regularizers import l2
from keras.layers import Input, Dense, GaussianNoise, Convolution3D, Flatten, Dropout, merge
from keras.models import Model

from cnn.keras.models.deepROI4.model_conv import build_model as mod


def build_model(input_shape=(1, 22, 22, 22), name='deepROI4'):
    name = name

    model_NC_in, model_NC_out = mod(input_shape=input_shape, name='deepROI6_NC')
    model_AD_in, model_AD_out = mod(input_shape=input_shape, name='deepROI6_AD')

    x = merge([model_NC_out, model_AD_out], mode='concat')

    x = Dense(512, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense1')(x)
    x = Dropout(0.5, name=name+'_dropout1')(x)
    x = Dense(256, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense2')(x)
    x = Dropout(0.5, name=name+'_dropout2')(x)
    x = Dense(128, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense3')(x)
    x = Dropout(0.5, name=name+'_dropout3')(x)

    output = Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), name=name+'_dense4')(x)

    return Model(input=[model_NC_in, model_AD_in], output=output)

