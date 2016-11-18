from keras.models import Sequential
from keras.layers.core import Dense


def build_model():
    model = Sequential()

    model.add(Dense(16, activation='relu', input_dim=3))
    model.add(Dense(16, activation='relu'))

    return model
