from keras.models import Sequential
from cnn.keras.models.meanROI1_1.model_single1_conv import build_model as mod1
from cnn.keras.models.meanROI1_2.model_single1_conv import build_model as mod2
from cnn.keras.models.meanROI1_3.model_single1_conv import build_model as mod3
from cnn.keras.models.meanROI1_4.model_single1_conv import build_model as mod4
from cnn.keras.models.meanROI1_5.model_single1_conv import build_model as mod5
from cnn.keras.meanROI1.diff_model import build_model as mod_diff
from keras.layers import Merge
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2


def build_model(num_classes):
    model1 = mod1()
    model2 = mod2()
    model3 = mod3()
    model4 = mod4()
    model5 = mod5()
    model_diff = mod_diff(5)

    merged = Merge([model1, model2, model3, model4, model5, model_diff], mode='concat')

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

