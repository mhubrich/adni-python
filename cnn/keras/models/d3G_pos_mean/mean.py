from keras.models import Sequential
from keras.layers.core import Dense


def build_model():
    model = Sequential()

    model.add(Dense(1, activation='relu', input_dim=1))

    return model
