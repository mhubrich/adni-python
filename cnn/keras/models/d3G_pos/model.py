from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout
from cnn.keras.models.d3G_pos.d3G import build_model as mod1
from cnn.keras.models.position.model import build_model as mod2

def build_model(num_classes, input_shape=(1, 29, 29, 29)):
    model1 = mod1(input_shape)
    model2 = mod2()

    merged = Merge([model1, model2], mode='concat')

    model = Sequential()
    model.add(merged)

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(512, activation='relu'))  # new
    #model.add(Dropout(0.5))  # new
    model.add(Dense(num_classes, activation='softmax'))

    return model
