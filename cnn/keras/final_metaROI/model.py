from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, merge
from keras.models import Model

from cnn.keras.final_metaROI.model21_conv import build_model as mod1
from cnn.keras.final_metaROI.model22_conv import build_model as mod2
from cnn.keras.final_metaROI.model23_conv import build_model as mod3
from cnn.keras.final_metaROI.model24_conv import build_model as mod4
from cnn.keras.final_metaROI.model25_conv import build_model as mod5


def build_model(trainable=True):
    name = 'metaROI'

    model1_in, model1_out = mod1(trainable=trainable)
    model2_in, model2_out = mod2(trainable=trainable)
    model3_in, model3_out = mod3(trainable=trainable)
    model4_in, model4_out = mod4(trainable=trainable)
    model5_in, model5_out = mod5(trainable=trainable)

    x = merge([model1_out, model2_out, model3_out, model4_out, model5_out], mode='concat')

    x = Dense(512, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense1')(x)
    x = Dropout(0.5, name=name+'_dropout1')(x)
    x = Dense(256, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense2')(x)
    x = Dropout(0.5, name=name+'_dropout2')(x)
    x = Dense(128, activation='relu', W_regularizer=l2(0.0001), name=name+'_dense3')(x)
    x = Dropout(0.5, name=name+'_dropout3')(x)

    output = Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), name=name+'_dense4')(x)

    return Model(input=[model1_in, model2_in, model3_in, model4_in, model5_in], output=output)

