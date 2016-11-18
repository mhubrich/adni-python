from keras.models import Model
from cnn.keras.models.metaROI1.model1_f import build_model as mod1
from cnn.keras.models.metaROI2.model1_f import build_model as mod2
from cnn.keras.models.metaROI3.model1_f import build_model as mod3
from cnn.keras.models.metaROI4.model1_f import build_model as mod4
from cnn.keras.models.metaROI5.model1_f import build_model as mod5
from cnn.keras.metaROI_autoencoder.model_input import build_model as mod_auto
from keras.layers import Dense, merge, Dropout
from keras.regularizers import l2


def build_model(num_classes):
    input1, input11, model1 = mod1()
    input2, input22, model2 = mod2()
    input3, input33, model3 = mod3()
    input4, input44, model4 = mod4()
    input5, input55, model5 = mod5()
    #model_auto_1 = mod_auto()
    #model_auto_2 = mod_auto()
    #model_auto_3 = mod_auto()
    #model_auto_4 = mod_auto()
    #model_auto_5 = mod_auto()

    #x = merge([model1, model_auto_1, model2, model_auto_2, model3, model_auto_3, model4, model_auto_4, model5, model_auto_5], mode='concat')
    x = merge([model1, model2, model3, model4, model5], mode='concat')

    #x = Dropout(0.1)(x)
    #x = Dense(1024, activation='relu', W_regularizer=l2(0.000))(x)
    #x = Dropout(0.1)(x)
    x = Dense(512, activation='relu', W_regularizer=l2(0.000))(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu', W_regularizer=l2(0.000))(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', W_regularizer=l2(0.000))(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', W_regularizer=l2(0.000))(x)
    x = Dropout(0.1)(x)
    out = Dense(num_classes, activation='softmax', W_regularizer=l2(0.000))(x)

    return Model(input=[input1, input11, input2, input22, input3, input33, input4, input44, input5, input55], output=out)
