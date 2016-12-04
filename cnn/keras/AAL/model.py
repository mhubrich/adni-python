from keras.models import Sequential
from cnn.keras.models.AAL64.model_single4_conv import build_model as mod1
from cnn.keras.models.AAL65.model_single5_conv import build_model as mod2
from cnn.keras.models.AAL34.model_single7_conv import build_model as mod3
from cnn.keras.models.AAL35.model_single8_conv import build_model as mod4
from cnn.keras.models.AAL61.model_single5_conv import build_model as mod5
#from cnn.keras.models.meanROI2_6.model_single1_conv import build_model as mod6
from cnn.keras.AAL.diff_model import build_model as mod_diff
from keras.layers import Merge
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2


def build_model(num_classes):
    model1 = mod1()
    model2 = mod2()
    model3 = mod3()
    model4 = mod4()
    model5 = mod5()
    #model6 = mod6()
    model_diff = mod_diff()

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

