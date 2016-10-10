from keras.models import Sequential
from cnn.keras.models.partitions2 import d3G
from keras.layers import Merge
from keras.layers.core import Dense, Dropout


def build_model(num_classes, input_shape=(1, 29, 29, 29)):
    model1 = d3G.build_model(input_shape)
    model2 = d3G.build_model(input_shape)
    model3 = d3G.build_model(input_shape)

    merged = Merge([model1, model2, model3], mode='concat')

    model = Sequential()
    model.add(merged)

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
