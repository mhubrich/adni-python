from keras.models import Sequential
from cnn.keras.models.deepROI2.model_NC_conv import build_model as modNC
from cnn.keras.models.deepROI2.model_AD_conv import build_model as modAD
from cnn.keras.models.AVG444.model_conv import build_model as mod
from keras.layers import Merge
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2


def build_model(num_classes=1):
    model1 = modNC()
    model2 = modAD()
    model3 = mod()

    merged = Merge([model1, model2, model3], mode='concat')

    model = Sequential()
    model.add(merged)

    model.add(Dense(512, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    if num_classes > 1:
        model.add(Dense(num_classes, activation='softmax', W_regularizer=l2(0.0001)))
    else:
        model.add(Dense(num_classes, activation='sigmoid', W_regularizer=l2(0.0001)))
    return model

